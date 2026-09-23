"""The 'Guarded Move / Guarded Trace' loop from the control-flow diagram: advance toward
a target in small steps, checking the obstacle detector on every single step, and re-
routing around anything in the way via one shared detour routine. Both the pre-grasp
approach and the mid-trace following call this same function - see esp32_dev_brief.md's
design notes on why there's only one replanner instead of two.
"""

import logging
import math
import time

log = logging.getLogger("planner")


def _lerp(a, b, t):
    return a + (b - a) * t


def _replan_detour(current_xy, final_target_xy, obstacle_world_xy, clearance_mm):
    dx = final_target_xy[0] - current_xy[0]
    dy = final_target_xy[1] - current_xy[1]
    length = math.hypot(dx, dy) or 1.0
    perp = (-dy / length, dx / length)
    return (
        obstacle_world_xy[0] + perp[0] * clearance_mm,
        obstacle_world_xy[1] + perp[1] * clearance_mm,
    )


def guarded_move(arm_link, ik_solver, obstacle_detector, camera, homography,
                  current_xy, final_target_xy, z_mm, gripper_deg,
                  step_mm=5.0, clearance_mm=30.0, max_steps=500, on_override_wait=None):
    """Returns the (x, y) reached - always final_target_xy unless it raises."""
    for _ in range(max_steps):
        if math.hypot(final_target_xy[0] - current_xy[0], final_target_xy[1] - current_xy[1]) < step_mm:
            _send(arm_link, ik_solver, final_target_xy, z_mm, gripper_deg)
            return final_target_xy

        frame = camera.read()
        obstacle_px = obstacle_detector.detect(frame)
        if obstacle_px is not None:
            ox, oy, ow, oh = obstacle_px
            obstacle_world = homography.pixel_to_world(ox + ow / 2, oy + oh / 2)
            log.warning(f"obstacle at {obstacle_world} - replanning detour")
            aim_at = _replan_detour(current_xy, final_target_xy, obstacle_world, clearance_mm)
        else:
            aim_at = final_target_xy  # clear path - head straight for the real goal again

        while arm_link.override_active:
            if on_override_wait:
                on_override_wait()
            time.sleep(0.2)

        dist = math.hypot(aim_at[0] - current_xy[0], aim_at[1] - current_xy[1]) or 1.0
        t = min(1.0, step_mm / dist)
        current_xy = (_lerp(current_xy[0], aim_at[0], t), _lerp(current_xy[1], aim_at[1], t))
        _send(arm_link, ik_solver, current_xy, z_mm, gripper_deg)

    raise RuntimeError(f"guarded_move did not reach {final_target_xy} within {max_steps} steps")


def _send(arm_link, ik_solver, xy, z_mm, gripper_deg):
    base, shoulder, elbow = ik_solver(xy[0], xy[1], z_mm)
    arm_link.move(base, shoulder, elbow, gripper_deg)
