"""Locate the marker/pen by color. Deterministic classical CV on purpose - a missed
grasp is worse than an AI flourish here, and this needs zero training data to work."""

import cv2
import numpy as np


def find_marker(frame, hsv_lower, hsv_upper, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None

    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]
    return (u, v)
