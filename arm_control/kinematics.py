"""Analytical IK for the arm: base yaw + a 2-link planar solve (shoulder/elbow) in the
vertical plane through the target. No Jacobian/numerical solver needed - closed-form is
fast enough to call every guarded-move step, and deterministic, which matters since
replanning calls this constantly.

Confirmed from the real hardware (schematic + teammate): there are only 3 positioning
servos - base (S1), shoulder (S2), elbow (S3) - plus a 4th servo (S4) that is purely the
gripper, not a wrist-tilt joint. There is no active orientation control: whether the pen
stays vertical across the workspace depends on whether this arm's upper links form a
passive parallel four-bar linkage (mechanically self-leveling) or are simple rigid links
(orientation will drift with reach) - confirm which before trusting drawing quality far
from wherever you calibrated pen-verticality by eye.
"""

import math


def solve_ik(x_mm, y_mm, z_mm, link1_mm, link2_mm, base_height_mm):
    """Returns (base_deg, shoulder_deg, elbow_deg). Raises ValueError if the target is
    out of reach - let it propagate; a reachability failure means your path or workspace
    calibration put a waypoint outside the arm's physical envelope."""
    base_deg = math.degrees(math.atan2(y_mm, x_mm))

    r = math.hypot(x_mm, y_mm)          # horizontal distance from the base's yaw axis
    dz = z_mm - base_height_mm          # vertical offset from the shoulder pivot to the target
    reach = math.hypot(r, dz)

    max_reach = link1_mm + link2_mm
    if reach > max_reach:
        raise ValueError(f"target ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) needs reach "
                          f"{reach:.1f}mm > max {max_reach:.1f}mm")

    cos_elbow = (reach ** 2 - link1_mm ** 2 - link2_mm ** 2) / (2 * link1_mm * link2_mm)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_rad = math.acos(cos_elbow)

    shoulder_rad = math.atan2(dz, r) - math.atan2(
        link2_mm * math.sin(elbow_rad),
        link1_mm + link2_mm * math.cos(elbow_rad),
    )
    shoulder_deg = math.degrees(shoulder_rad)
    elbow_deg = math.degrees(elbow_rad)

    return base_deg, shoulder_deg, elbow_deg
