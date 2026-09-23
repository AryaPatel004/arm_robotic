import json

import cv2
import numpy as np


class Homography:
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.pixel_points = np.array(data["pixel_points"], dtype=np.float32)
        self.world_points = np.array(data["world_points_mm"], dtype=np.float32)
        self.to_world_m = cv2.getPerspectiveTransform(self.pixel_points, self.world_points)
        self.to_pixel_m = cv2.getPerspectiveTransform(self.world_points, self.pixel_points)

    def pixel_to_world(self, u, v):
        pt = np.array([[[u, v]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.to_world_m)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def world_to_pixel(self, x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.to_pixel_m)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def worlds_to_pixels(self, pts):
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.to_pixel_m).reshape(-1, 2)

    def roi_mask(self, shape):
        mask = np.zeros(shape[:2], np.uint8)
        hull = cv2.convexHull(self.pixel_points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def world_circle_mask(self, shape, center_xy, radius_mm, mask=None):
        if mask is None:
            mask = np.zeros(shape[:2], np.uint8)
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([center_xy[0] + radius_mm * np.cos(a), center_xy[1] + radius_mm * np.sin(a)], axis=1)
        cv2.fillPoly(mask, [self.worlds_to_pixels(ring).astype(np.int32)], 255)
        return mask

    def world_corridor_mask(self, shape, a_xy, b_xy, width_mm, mask=None):
        if mask is None:
            mask = np.zeros(shape[:2], np.uint8)
        a = np.array(a_xy, np.float64)
        b = np.array(b_xy, np.float64)
        d = b - a
        n = np.hypot(*d)
        if n < 1e-6:
            return self.world_circle_mask(shape, a_xy, width_mm / 2, mask)
        p = np.array([-d[1], d[0]]) / n * width_mm / 2
        quad = np.array([a + p, b + p, b - p, a - p])
        cv2.fillPoly(mask, [self.worlds_to_pixels(quad).astype(np.int32)], 255)
        return mask
