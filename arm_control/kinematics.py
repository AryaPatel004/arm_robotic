import math

JOINT_NAMES = ("base", "shoulder", "elbow", "wrist")


def solve_ik(x_mm, y_mm, z_mm, link1_mm, link2_mm, base_height_mm, wrist_offset_deg=0.0):
    base_deg = math.degrees(math.atan2(y_mm, x_mm))

    r = math.hypot(x_mm, y_mm)
    dz = z_mm - base_height_mm
    reach = math.hypot(r, dz)

    max_reach = link1_mm + link2_mm
    min_reach = abs(link1_mm - link2_mm)
    if reach > max_reach or reach < min_reach:
        raise ValueError(f"target ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) needs reach "
                         f"{reach:.1f}mm, allowed {min_reach:.1f}-{max_reach:.1f}mm")

    cos_elbow = (reach ** 2 - link1_mm ** 2 - link2_mm ** 2) / (2 * link1_mm * link2_mm)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_rad = math.acos(cos_elbow)

    shoulder_rad = math.atan2(dz, r) - math.atan2(
        link2_mm * math.sin(elbow_rad),
        link1_mm + link2_mm * math.cos(elbow_rad),
    )
    shoulder_deg = math.degrees(shoulder_rad)
    elbow_deg = math.degrees(elbow_rad)
    wrist_deg = wrist_offset_deg - (shoulder_deg + elbow_deg)

    return base_deg, shoulder_deg, elbow_deg, wrist_deg


def to_servo(angles, zero_deg, sign, limits):
    out = []
    for name, a, z, s in zip(JOINT_NAMES, angles, zero_deg, sign):
        d = z + s * a
        lo, hi = limits[name]
        if d < lo or d > hi:
            raise ValueError(f"{name} needs servo {d:.0f}deg, allowed {lo}-{hi}")
        out.append(d)
    return tuple(out)


def make_ik_solver(cfg):
    def ik_solver(x, y, z):
        lx = x - cfg.BASE_XY_MM[0]
        ly = y - cfg.BASE_XY_MM[1]
        angles = solve_ik(lx, ly, z + cfg.TOOL_LENGTH_MM, cfg.LINK1_MM, cfg.LINK2_MM,
                          cfg.BASE_HEIGHT_MM, cfg.WRIST_OFFSET_DEG)
        return to_servo(angles, cfg.SERVO_ZERO_DEG, cfg.SERVO_SIGN, cfg.JOINT_LIMITS_DEG)
    return ik_solver
