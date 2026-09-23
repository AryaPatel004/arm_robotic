"""Pixel <-> world-mm mapping for the overhead camera looking at a flat work surface.
Since the arm effectively works in a plane, a 4-point homography stands in for full 3D
camera calibration - see calibrate.py to produce homography.json."""

import json

import cv2
import numpy as np


class Homography:
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        px = np.array(data["pixel_points"], dtype=np.float32)
        world = np.array(data["world_points_mm"], dtype=np.float32)
        self.to_world_m = cv2.getPerspectiveTransform(px, world)
        self.to_pixel_m = cv2.getPerspectiveTransform(world, px)

    def pixel_to_world(self, u, v):
        pt = np.array([[[u, v]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.to_world_m)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def world_to_pixel(self, x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.to_pixel_m)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
