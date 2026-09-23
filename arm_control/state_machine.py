"""Top-level reactive control loop implementing the flowchart: search marker, guarded-
move to it, grasp, guarded-move to the path start, guarded-trace the path, place the
marker back down, return home. No waypoint list is ever stored ahead of time - the
marker's location and the path's shape both come from the live camera on every run.
"""

import logging
import time

import config as cfg
from marker import find_marker
from path_trace import find_path_waypoints
from planner import guarded_move

log = logging.getLogger("state_machine")


class ArmController:
    def __init__(self, camera, homography, arm_link, obstacle_detector, ik_solver):
        self.camera = camera
        self.homography = homography
        self.arm_link = arm_link
        self.obstacle_detector = obstacle_detector
        self.ik_solver = ik_solver
        self.current_xy = cfg.HOME_XY_MM
        self.gripper_deg = cfg.GRIPPER_OPEN_DEG

    def run_once(self):
        log.info("START - homing")
        self.arm_link.home()
        self.current_xy = cfg.HOME_XY_MM
        self.gripper_deg = cfg.GRIPPER_OPEN_DEG

        marker_xy = self._search_marker()
        log.info(f"marker found at {marker_xy}")
        self._move_to(marker_xy, cfg.PEN_UP_Z_MM)

        self.gripper_deg = cfg.GRIPPER_CLOSED_DEG
        self._apply_gripper(cfg.PEN_UP_Z_MM)
        log.info("marker grasped")

        path = self._find_path()
        log.info(f"path has {len(path)} waypoints")

        self._move_to(path[0], cfg.PEN_DOWN_Z_MM)
        for waypoint in path[1:]:
            self._move_to(waypoint, cfg.PEN_DOWN_Z_MM)
        log.info("path traced")

        self._move_to(marker_xy, cfg.PEN_UP_Z_MM)
        self.gripper_deg = cfg.GRIPPER_OPEN_DEG
        self._apply_gripper(cfg.PEN_UP_Z_MM)
        log.info("marker placed back down")

        self.arm_link.home()
        self.current_xy = cfg.HOME_XY_MM
        log.info("IDLE - run complete")

    def _move_to(self, target_xy, z_mm):
        self.current_xy = guarded_move(
            self.arm_link, self.ik_solver, self.obstacle_detector, self.camera, self.homography,
            self.current_xy, target_xy, z_mm, self.gripper_deg,
            step_mm=cfg.STEP_MM, clearance_mm=cfg.DETOUR_CLEARANCE_MM,
            max_steps=cfg.MAX_STEPS_PER_MOVE, on_override_wait=self._wait_for_resume,
        )

    def _apply_gripper(self, z_mm):
        base, shoulder, elbow = self.ik_solver(self.current_xy[0], self.current_xy[1], z_mm)
        self.arm_link.move(base, shoulder, elbow, self.gripper_deg)

    def _search_marker(self, timeout_s=30):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame = self.camera.read()
            pixel = find_marker(frame, cfg.MARKER_HSV_LOWER, cfg.MARKER_HSV_UPPER, cfg.MARKER_MIN_AREA_PX)
            if pixel:
                return self.homography.pixel_to_world(*pixel)
            time.sleep(0.1)
        raise RuntimeError("marker not found within timeout")

    def _find_path(self, retries=10):
        marker_px = self.homography.world_to_pixel(*self.current_xy)
        for _ in range(retries):
            frame = self.camera.read()
            pixels = find_path_waypoints(frame, cfg.PATH_HSV_LOWER, cfg.PATH_HSV_UPPER, start_near_px=marker_px)
            if pixels:
                return [self.homography.pixel_to_world(u, v) for (u, v) in pixels]
            time.sleep(0.2)
        raise RuntimeError("path not found")

    def _wait_for_resume(self):
        log.warning("override active - waiting for release, then resuming")
        while self.arm_link.override_active:
            time.sleep(0.3)
        self.arm_link.resume()
