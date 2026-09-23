import json
import os

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS_MM_PATH = os.path.join(HERE, "markers_mm.json")
WORKSPACE_PATH = os.path.join(HERE, "workspace.json")
HOMOGRAPHY_PATH = os.path.join(HERE, "homography.json")
REFERENCE_FRAME_PATH = os.path.join(HERE, "reference_frame.png")

S_MIN = 90
V_MIN = 40
BROWN_V = 110
MIN_MARKER_PX = 400
MERGE_PX = 15
MIN_OBS_PX = 800
DIFF_T = 30

COLORS = {
    "green": [(35, 85)],
    "blue": [(85, 125)],
    "purple": [(125, 170)],
    "orange": [(10, 22)],
    "red": [(0, 10), (170, 180)],
    "yellow": [(22, 35)],
    "brown": [(0, 22)],
}
DRAW = {
    "green": (0, 200, 0), "blue": (255, 80, 0), "purple": (200, 0, 160), "orange": (0, 140, 255),
    "red": (0, 0, 255), "yellow": (0, 220, 255), "brown": (30, 60, 110),
}

K_OPEN = np.ones((5, 5), np.uint8)
K_MERGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_PX, MERGE_PX))
K_OBS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ARUCO = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())


def load_markers_mm(path=MARKERS_MM_PATH):
    try:
        return {int(k): v for k, v in json.load(open(path)).items()}
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def load_workspace(path=WORKSPACE_PATH):
    try:
        d = json.load(open(path))
        return {"H": np.array(d["H"], np.float64), "roi": np.array(d["roi_px"], np.int32), "resolution": d["resolution"]}
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def to_mm(ws, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), ws["H"]).reshape(-1, 2)


def roi_mask(ws, shape):
    if ws is None:
        return np.full(shape, 255, np.uint8)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [ws["roi"]], 255)
    return m


def prep_gray(f):
    return cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (7, 7), 0)


def color_mask(h, s, v, name):
    m = np.zeros(h.shape, bool)
    for lo, hi in COLORS[name]:
        m |= (h >= lo) & (h < hi)
    m &= s >= S_MIN
    if name == "brown":
        m &= (v >= V_MIN) & (v < BROWN_V)
    elif name in ("red", "orange"):
        m &= v >= BROWN_V
    else:
        m &= v >= V_MIN
    return cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, K_OPEN)


def blob_info(c, ws):
    mo = cv2.moments(c)
    if mo["m00"] == 0:
        return None
    px = (int(mo["m10"] / mo["m00"]), int(mo["m01"] / mo["m00"]))
    ctr = np.array([[mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]]], np.float32)
    pts = c.reshape(-1, 2).astype(np.float32)
    if ws is not None:
        ctr = to_mm(ws, ctr)
        pts = to_mm(ws, pts)
    p = pts.astype(np.float64)
    p -= p.mean(axis=0)
    _, vec = np.linalg.eigh(np.cov(p.T))
    ang = float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) % 180)
    return {"x": round(float(ctr[0, 0]), 1), "y": round(float(ctr[0, 1]), 1), "angle": round(ang, 1), "px": list(px)}


def hsv_stats(h, s, v, mask):
    sel = mask > 0
    if not sel.any():
        return None
    hh, ss, vv = h[sel], s[sel], v[sel]
    lo = [int(np.percentile(hh, 5)), int(np.percentile(ss, 5)), int(np.percentile(vv, 5))]
    hi = [int(np.percentile(hh, 95)), int(np.percentile(ss, 95)), int(np.percentile(vv, 95))]
    return {"lower": [max(0, lo[0] - 5), max(0, lo[1] - 30), max(0, lo[2] - 30)],
            "upper": [min(179, hi[0] + 5), 255, 255]}


def find_markers(h, s, v, ws, roi):
    out = []
    for name in COLORS:
        m = color_mask(h, s, v, name) & roi
        merged = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K_MERGE)
        cs, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if cv2.contourArea(c) < MIN_MARKER_PX:
                continue
            info = blob_info(c, ws)
            if info is None:
                continue
            info["color"] = name
            blob = np.zeros(h.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            info["hsv"] = hsv_stats(h, s, v, blob & m)
            out.append((info, c))
    return out


def aruco_find(frame):
    corners, ids, _ = ARUCO.detectMarkers(frame)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found[int(i)] = c[0].mean(axis=0)
    return found, corners, ids


def save_calibration(found, mm_cfg, shape, path=WORKSPACE_PATH):
    if len(mm_cfg) < 4 or any(i not in found for i in mm_cfg):
        return None
    order = sorted(mm_cfg)
    px = np.array([found[i] for i in order], np.float32)
    world = np.array([mm_cfg[i] for i in order], np.float32)
    H, _ = cv2.findHomography(px, world)
    if H is None:
        return None
    with open(path, "w") as fh:
        json.dump({"H": H.tolist(), "roi_px": px.tolist(), "resolution": [shape[1], shape[0]]}, fh, indent=2)
    with open(HOMOGRAPHY_PATH, "w") as fh:
        json.dump({"pixel_points": px[:4].tolist(), "world_points_mm": world[:4].tolist()}, fh, indent=2)
    return load_workspace(path)


def detect(h, s, v, gray, bg, ws, roi, target):
    tm = cv2.morphologyEx(color_mask(h, s, v, target) & roi, cv2.MORPH_CLOSE, K_MERGE)
    cs, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= MIN_MARKER_PX]
    pen, pen_c = None, None
    if cs:
        pen_c = max(cs, key=cv2.contourArea)
        pen = blob_info(pen_c, ws)
        if pen is not None:
            pen["color"] = target
    _, d = cv2.threshold(cv2.absdiff(gray, bg), DIFF_T, 255, cv2.THRESH_BINARY)
    d &= roi
    d &= cv2.bitwise_not(cv2.dilate(tm, K_OBS, iterations=4))
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, K_OBS)
    d = cv2.dilate(d, K_OBS, iterations=2)
    cs, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obs, hulls = [], []
    for c in cs:
        if cv2.contourArea(c) < MIN_OBS_PX:
            continue
        hull = cv2.convexHull(c)
        pts = hull.reshape(-1, 2).astype(np.float32)
        if ws is not None:
            pts = to_mm(ws, pts)
        (x, y), r = cv2.minEnclosingCircle(pts)
        obs.append({"x": round(x, 1), "y": round(y, 1), "r": round(r, 1)})
        hulls.append(hull)
    return pen, pen_c, obs, hulls
