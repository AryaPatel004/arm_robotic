"""Extract an ordered sequence of waypoints from a single drawn stroke on the work
surface. Assumes one open curve with no branches or loops, which matches "trace a path
from start to finish" - not a general line-drawing graph parser.

Requires opencv-contrib-python (for cv2.ximgproc.thinning), not plain opencv-python.
"""

import cv2
import numpy as np


def _find_endpoints(skeleton):
    pts = [tuple(p) for p in np.argwhere(skeleton > 0)]
    coords = set(pts)
    endpoints = []
    for y, x in pts:
        neighbors = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in coords:
                    neighbors += 1
        if neighbors == 1:
            endpoints.append((y, x))
    return endpoints


def _walk_chain(skeleton, start):
    coords = set(tuple(p) for p in np.argwhere(skeleton > 0))
    visited = {start}
    chain = [start]
    current = start
    while True:
        y, x = current
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                candidate = (y + dy, x + dx)
                if candidate in coords and candidate not in visited:
                    nxt = candidate
                    break
            if nxt:
                break
        if nxt is None:
            break
        visited.add(nxt)
        chain.append(nxt)
        current = nxt
    return chain


def find_path_waypoints(frame, lower, upper, start_near_px=None, min_length_px=40, sample_every=12):
    """Returns an ordered list of (u, v) pixel waypoints from start to finish, or None
    if no clean single-stroke path was found. If start_near_px is given, the chain starts
    from whichever endpoint is closest to it (e.g. the marker's current position)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    try:
        skeleton = cv2.ximgproc.thinning(mask)
    except AttributeError as exc:
        raise RuntimeError(
            "cv2.ximgproc.thinning is missing - install opencv-contrib-python, not opencv-python"
        ) from exc

    endpoints = _find_endpoints(skeleton)
    if len(endpoints) < 2:
        return None

    if start_near_px is not None:
        sx, sy = start_near_px
        start = min(endpoints, key=lambda p: (p[1] - sx) ** 2 + (p[0] - sy) ** 2)
    else:
        start = endpoints[0]

    chain = _walk_chain(skeleton, start)
    if len(chain) < min_length_px:
        return None

    waypoints = [(x, y) for (y, x) in chain[::sample_every]]
    last_y, last_x = chain[-1]
    if waypoints[-1] != (last_x, last_y):
        waypoints.append((last_x, last_y))
    return waypoints
