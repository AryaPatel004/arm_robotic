import logging
import math
import time

import config as cfg
from marker import find_marker
from path_trace import find_path_waypoints
from planner import GuardedMover

log = logging.getLogger("state_machine")


class ArmController:
    def __init__(self, camera, homography, arm_link, obstacle_detector, ik_solver):
        self.camera = camera
        self.homography = homography
        self.arm_link = arm_link
        self.ik_solver = ik_solver
        self.current_xy = cfg.HOME_XY_MM
        self.gripper_deg = cfg.GRIPPER_OPEN_DEG
        frame = camera.read()
        self.roi = homography.roi_mask(frame.shape)
        obstacle_detector.set_roi(self.roi)
        self.mover = GuardedMover(arm_link, ik_solver, obstacle_detector, camera, homography, cfg,
                                  on_override_wait=self._wait_for_resume)
        self.state = "IDLE"
        hx, hy = homography.world_to_pixel(*cfg.HOME_XY_MM)
        if 0 <= int(hy) < self.roi.shape[0] and 0 <= int(hx) < self.roi.shape[1] and self.roi[int(hy), int(hx)]:
            log.warning("HOME_XY_MM is inside the work area - the parked arm will hide part of the path from the camera")

    def _set_state(self, s):
        self.state = s
        log.info(f"STATE {s}")

    def run_once(self):
        self._set_state("HOME")
        self.arm_link.home()
        self.arm_link.wait_settled()
        self.gripper_deg = cfg.GRIPPER_OPEN_DEG
        self.current_xy = cfg.HOME_XY_MM
        self.mover.current_z = None
        self.mover.goto_direct(self.current_xy, cfg.PEN_UP_Z_MM, self.gripper_deg)

        self._set_state("SEARCH_MARKER")
        marker_xy = self._search_marker()
        log.info(f"marker at ({marker_xy[0]:.0f},{marker_xy[1]:.0f}) mm")
        self.mover.add_exclusion(marker_xy, cfg.PICKUP_MASK_RADIUS_MM)

        self._set_state("FIND_PATH")
        path = self._find_path(marker_xy)
        log.info(f"path: {len(path)} waypoints from ({path[0][0]:.0f},{path[0][1]:.0f}) to ({path[-1][0]:.0f},{path[-1][1]:.0f})")
        self._check_reach(path, marker_xy)

        self._set_state("APPROACH_MARKER")
        self._travel(marker_xy)

        self._set_state("GRASP")
        self.mover.goto_direct(marker_xy, cfg.GRASP_Z_MM, cfg.GRIPPER_OPEN_DEG)
        self.gripper_deg = cfg.GRIPPER_CLOSED_DEG
        self.mover.goto_direct(marker_xy, cfg.GRASP_Z_MM, self.gripper_deg)
        self.mover.goto_direct(marker_xy, cfg.PEN_UP_Z_MM, self.gripper_deg)

        self._set_state("GOTO_PATH_START")
        self._travel(path[0])

        self._set_state("TRACE")
        self.mover.goto_direct(path[0], cfg.PEN_DOWN_Z_MM, self.gripper_deg)
        self._trace(path)
        self.mover.goto_direct(self.current_xy, cfg.PEN_UP_Z_MM, self.gripper_deg)

        self._set_state("RETURN_MARKER")
        self._travel(marker_xy)
        self.mover.goto_direct(marker_xy, cfg.GRASP_Z_MM, self.gripper_deg)
        self.gripper_deg = cfg.GRIPPER_OPEN_DEG
        self.mover.goto_direct(marker_xy, cfg.GRASP_Z_MM, self.gripper_deg)
        self.mover.goto_direct(marker_xy, cfg.PEN_UP_Z_MM, self.gripper_deg)
        self.mover.remove_exclusion(marker_xy)

        self._set_state("RETURN_HOME")
        self._travel(cfg.HOME_XY_MM)
        self.arm_link.home()
        self.arm_link.wait_settled()
        self._set_state("IDLE")

    def _blocked_index(self, path, start, obstacles, margin_mm, horizon_mm):
        travelled = 0.0
        prev = self.current_xy
        for j in range(start, len(path)):
            p = path[j]
            travelled += math.hypot(p[0] - prev[0], p[1] - prev[1])
            for ox, oy, r in obstacles:
                if math.hypot(p[0] - ox, p[1] - oy) < r + margin_mm:
                    return j
            if travelled > horizon_mm:
                return None
            prev = p
        return None

    def _clear_after(self, path, j, obstacles):
        for k in range(j + 1, len(path)):
            p = path[k]
            if all(math.hypot(p[0] - ox, p[1] - oy) >= r + cfg.DETOUR_CLEARANCE_MM for ox, oy, r in obstacles):
                return k
        return None

    def _trace(self, path):
        i = 1
        waits = 0
        while i < len(path):
            frame = self.camera.read()
            obstacles = self.mover.obstacles(frame, self.arm_xy())
            j = self._blocked_index(path, i, obstacles, cfg.PATH_CORRIDOR_MM, cfg.TRACE_LOOKAHEAD_MM)
            if j is None:
                self.current_xy = self.mover.move(self.current_xy, path[i], cfg.PEN_DOWN_Z_MM, self.gripper_deg,
                                                  detour_z_mm=cfg.PEN_UP_Z_MM)
                i += 1
                waits = 0
                continue
            k = self._clear_after(path, j, obstacles) if cfg.TRACE_BLOCK_MODE == "skip" else None
            if k is None:
                if waits == 0:
                    log.warning("obstacle on the path ahead - holding pen up until it moves")
                    self.mover.goto_direct(self.current_xy, cfg.PEN_UP_Z_MM, self.gripper_deg)
                waits += 1
                if waits * cfg.BLOCKED_WAIT_S > 120:
                    raise RuntimeError("path blocked for 2 minutes")
                time.sleep(cfg.BLOCKED_WAIT_S)
                continue
            ox, oy, r = min(obstacles, key=lambda o: math.hypot(path[j][0] - o[0], path[j][1] - o[1]))
            log.info(f"obstacle on path at ({ox:.0f},{oy:.0f}) r={r:.0f}mm - lifting pen, skipping waypoints {j}-{k - 1}, "
                     f"rejoining at ({path[k][0]:.0f},{path[k][1]:.0f})")
            self.mover.goto_direct(self.current_xy, cfg.PEN_UP_Z_MM, self.gripper_deg)
            self._travel(path[k])
            self.mover.goto_direct(path[k], cfg.PEN_DOWN_Z_MM, self.gripper_deg)
            i = k + 1
            waits = 0

    def _travel(self, target_xy):
        self.current_xy = self.mover.move(self.current_xy, target_xy, cfg.PEN_UP_Z_MM, self.gripper_deg)

    def arm_xy(self):
        return self.mover.current_xy or self.current_xy

    def _arm_mask(self, shape):
        return self.mover.exclude_mask(shape, self.arm_xy())

    def _search_marker(self, timeout_s=30):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame = self.camera.read()
            pixel = find_marker(frame, cfg.MARKER_HSV_LOWER, cfg.MARKER_HSV_UPPER, cfg.MARKER_MIN_AREA_PX,
                                roi_mask=self.roi, exclude_mask=self._arm_mask(frame.shape))
            if pixel:
                return self.homography.pixel_to_world(*pixel)
            time.sleep(0.1)
        raise RuntimeError("marker not found within timeout - check MARKER_HSV_* in config.py")

    def _find_path(self, marker_xy, retries=10):
        marker_px = self.homography.world_to_pixel(*marker_xy)
        for _ in range(retries):
            frame = self.camera.read()
            exclude = self._arm_mask(frame.shape)
            self.homography.world_circle_mask(frame.shape, marker_xy, cfg.PICKUP_MASK_RADIUS_MM, exclude)
            pixels = find_path_waypoints(frame, cfg.PATH_HSV_LOWER, cfg.PATH_HSV_UPPER, start_near_px=marker_px,
                                         min_length_px=cfg.PATH_MIN_LENGTH_PX, sample_every=cfg.PATH_SAMPLE_EVERY_PX,
                                         roi_mask=self.roi, exclude_mask=exclude)
            if pixels:
                return [self.homography.pixel_to_world(u, v) for (u, v) in pixels]
            time.sleep(0.2)
        raise RuntimeError("path not found - check PATH_HSV_* in config.py and that the line is inside the markers")

    def _check_reach(self, path, marker_xy):
        checks = [(marker_xy, cfg.GRASP_Z_MM, "marker grasp"), (marker_xy, cfg.PEN_UP_Z_MM, "marker lift")]
        checks += [(p, cfg.PEN_DOWN_Z_MM, f"path point {i}") for i, p in enumerate(path)]
        for xy, z, name in checks:
            try:
                self.ik_solver(xy[0], xy[1], z)
            except ValueError as exc:
                raise RuntimeError(f"{name} at ({xy[0]:.0f},{xy[1]:.0f}) is out of reach: {exc}") from exc

    def _wait_for_resume(self):
        prev = self.state
        self._set_state("OVERRIDE")
        while self.arm_link.override_active:
            time.sleep(0.2)
        time.sleep(0.2)
        self.arm_link.resume()
        self._set_state(prev)
