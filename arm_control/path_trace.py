from collections import deque

import cv2
import numpy as np

NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _zhang_suen(mask):
    img = (mask > 0).astype(np.uint8)
    img = np.pad(img, 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(np.int32) for n in nb)
            seq = nb + [p2]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            core = img[1:-1, 1:-1]
            kill = (core == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if kill.any():
                core[kill] = 0
                changed = True
    return (img[1:-1, 1:-1] * 255).astype(np.uint8)


def _thin(mask):
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return ximgproc.thinning(mask)
    return _zhang_suen(mask)


def _bfs(pixels, start):
    parent = {start: None}
    dist = {start: 0}
    far = start
    q = deque([start])
    while q:
        p = q.popleft()
        for dy, dx in NEIGHBORS:
            n = (p[0] + dy, p[1] + dx)
            if n in pixels and n not in parent:
                parent[n] = p
                dist[n] = dist[p] + 1
                if dist[n] > dist[far]:
                    far = n
                q.append(n)
    return far, parent


def _longest_chain(pixels):
    a, _ = _bfs(pixels, next(iter(pixels)))
    b, parent = _bfs(pixels, a)
    chain = []
    p = b
    while p is not None:
        chain.append(p)
        p = parent[p]
    return chain


def path_mask(frame, lower, upper, roi_mask=None, exclude_mask=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)
    if exclude_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_path_waypoints(frame, lower, upper, start_near_px=None, min_length_px=40, sample_every=12,
                        roi_mask=None, exclude_mask=None):
    mask = path_mask(frame, lower, upper, roi_mask, exclude_mask)
    if not mask.any():
        return None
    skeleton = _thin(mask)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if n <= 1:
        return None
    counts = np.bincount(labels[skeleton > 0], minlength=n)
    counts[0] = 0
    best = int(np.argmax(counts))
    if counts[best] < min_length_px:
        return None

    ys, xs = np.nonzero((skeleton > 0) & (labels == best))
    pixels = set(zip(ys.tolist(), xs.tolist()))
    chain = _longest_chain(pixels)
    if len(chain) < min_length_px:
        return None

    if start_near_px is not None:
        sx, sy = start_near_px
        d_first = (chain[0][1] - sx) ** 2 + (chain[0][0] - sy) ** 2
        d_last = (chain[-1][1] - sx) ** 2 + (chain[-1][0] - sy) ** 2
        if d_last < d_first:
            chain.reverse()

    waypoints = [(x, y) for (y, x) in chain[::sample_every]]
    last_y, last_x = chain[-1]
    if waypoints[-1] != (last_x, last_y):
        waypoints.append((last_x, last_y))
    return waypoints
