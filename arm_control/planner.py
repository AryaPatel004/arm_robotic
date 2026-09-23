import logging
import math
import time

from arm_link import OverrideActiveError

log = logging.getLogger("planner")


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return _dist(p, a), 0.0
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return _dist(p, (a[0] + t * dx, a[1] + t * dy)), t


class GuardedMover:
    def __init__(self, arm_link, ik_solver, obstacle_detector, camera, homography, cfg, on_override_wait=None):
        self.arm = arm_link
        self.ik = ik_solver
        self.detector = obstacle_detector
        self.camera = camera
        self.h = homography
        self.cfg = cfg
        self.on_override_wait = on_override_wait
        self.exclusions = []
        self.current_z = None
        self.current_xy = None
        self.last_obstacles = []

    def add_exclusion(self, xy, radius_mm):
        self.exclusions.append((tuple(xy), radius_mm))

    def remove_exclusion(self, xy):
        self.exclusions = [e for e in self.exclusions if _dist(e[0], xy) > 1.0]

    def exclude_mask(self, shape, arm_xy):
        m = self.h.world_corridor_mask(shape, self.cfg.BASE_XY_MM, arm_xy, self.cfg.ARM_MASK_WIDTH_MM)
        self.h.world_circle_mask(shape, arm_xy, self.cfg.ARM_TIP_MASK_RADIUS_MM, m)
        for xy, r in self.exclusions:
            self.h.world_circle_mask(shape, xy, r, m)
        return m

    def obstacles(self, frame, arm_xy):
        boxes = self.detector.detect_all(frame, self.exclude_mask(frame.shape, arm_xy))
        out = []
        for x, y, w, h in boxes:
            corners = [self.h.pixel_to_world(u, v) for u, v in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
            cx = sum(c[0] for c in corners) / 4
            cy = sum(c[1] for c in corners) / 4
            r = max(_dist(c, (cx, cy)) for c in corners)
            out.append((cx, cy, r))
        self.last_obstacles = out
        return out

    def _reachable(self, xy, z):
        try:
            self.ik(xy[0], xy[1], z)
            return True
        except ValueError:
            return False

    def _send(self, xy, z, gripper_deg):
        base, shoulder, elbow, wrist = self.ik(xy[0], xy[1], z)
        for _ in range(3):
            self._wait_override()
            try:
                self.arm.move(base, shoulder, elbow, wrist, gripper_deg)
                self.current_z = z
                self.current_xy = tuple(xy)
                return
            except OverrideActiveError:
                continue
        raise RuntimeError("arm kept rejecting moves with OVERRIDE_ACTIVE")

    def _wait_override(self):
        if self.arm.override_active:
            if self.on_override_wait:
                self.on_override_wait()
            else:
                while self.arm.override_active:
                    time.sleep(0.2)

    def _pace(self):
        lead = self.arm.busy_s() - 0.1
        if lead > 0:
            time.sleep(lead)

    def goto_direct(self, xy, z, gripper_deg):
        for _ in range(5):
            before = self.arm.override_count
            self._send(xy, z, gripper_deg)
            self.arm.wait_settled()
            if self.arm.override_count == before and not self.arm.override_active:
                return
        raise RuntimeError("move kept getting interrupted by the kill switch")

    def _blocking(self, obstacles, a, b):
        best = None
        for ox, oy, r in obstacles:
            d, t = _seg_dist((ox, oy), a, b)
            if d < r + self.cfg.PATH_CORRIDOR_MM and (best is None or t < best[1]):
                best = ((ox, oy, r), t)
        return best[0] if best else None

    def _detour(self, current, target, obs):
        ox, oy, r = obs
        dx, dy = target[0] - current[0], target[1] - current[1]
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        left = (-uy, ux)
        cross = dx * (oy - current[1]) - dy * (ox - current[0])
        sides = [(-left[0], -left[1]), left] if cross > 0 else [left, (-left[0], -left[1])]
        off = r + self.cfg.DETOUR_CLEARANCE_MM
        for px, py in sides:
            cand = (ox + px * off + ux * r * 0.5, oy + py * off + uy * r * 0.5)
            if self._reachable(cand, self.current_z if self.current_z is not None else 0.0):
                return cand
        return None

    def move(self, current_xy, target_xy, z_mm, gripper_deg, detour_z_mm=None):
        cfg = self.cfg
        if self.current_z is None:
            self.current_z = z_mm
        if abs(self.current_z - z_mm) > 0.5 and detour_z_mm is None:
            self.goto_direct(current_xy, z_mm, gripper_deg)
        waits = 0
        steps = 0
        blocked_logged = False
        detouring = False
        while steps < cfg.MAX_STEPS_PER_MOVE:
            if _dist(current_xy, target_xy) < cfg.STEP_MM:
                if abs(self.current_z - z_mm) > 0.5:
                    self.goto_direct(current_xy, z_mm, gripper_deg)
                self._send(target_xy, z_mm, gripper_deg)
                return target_xy

            frame = self.camera.read()
            obstacles = self.obstacles(frame, current_xy)

            on_target = [o for o in obstacles if _dist((o[0], o[1]), target_xy) < o[2] + cfg.DETOUR_CLEARANCE_MM * 0.5]
            if on_target:
                if not blocked_logged:
                    log.warning(f"target {tuple(round(v) for v in target_xy)} is covered by an obstacle - holding")
                    blocked_logged = True
                waits += 1
                if waits * cfg.BLOCKED_WAIT_S > 120:
                    raise RuntimeError("target blocked for 2 minutes")
                time.sleep(cfg.BLOCKED_WAIT_S)
                continue
            if blocked_logged:
                log.info("target clear again - continuing")
                blocked_logged = False

            blocker = self._blocking(obstacles, current_xy, target_xy)
            want_z = z_mm
            if blocker is None:
                aim = target_xy
                if detouring:
                    log.info("path clear - heading straight for the target")
                    detouring = False
            else:
                aim = self._detour(current_xy, target_xy, blocker)
                if aim is None:
                    log.warning("obstacle in the way and no reachable detour - holding")
                    time.sleep(cfg.BLOCKED_WAIT_S)
                    waits += 1
                    continue
                if not detouring:
                    log.info(f"obstacle at ({blocker[0]:.0f},{blocker[1]:.0f}) r={blocker[2]:.0f}mm - detouring via ({aim[0]:.0f},{aim[1]:.0f})")
                    detouring = True
                if detour_z_mm is not None:
                    want_z = detour_z_mm
                if _dist(current_xy, aim) < cfg.STEP_MM:
                    aim = target_xy

            if abs(self.current_z - want_z) > 0.5:
                self.goto_direct(current_xy, want_z, gripper_deg)

            d = _dist(current_xy, aim) or 1.0
            t = min(1.0, cfg.STEP_MM / d)
            nxt = (current_xy[0] + (aim[0] - current_xy[0]) * t, current_xy[1] + (aim[1] - current_xy[1]) * t)
            try:
                self._send(nxt, want_z, gripper_deg)
            except ValueError as exc:
                raise RuntimeError(f"step to ({nxt[0]:.0f},{nxt[1]:.0f}) unreachable: {exc}") from exc
            current_xy = nxt
            steps += 1
            self._pace()

        raise RuntimeError(f"did not reach {target_xy} within {cfg.MAX_STEPS_PER_MOVE} steps")
