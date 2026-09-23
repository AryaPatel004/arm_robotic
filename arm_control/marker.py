import cv2
import numpy as np


def find_marker(frame, hsv_lower, hsv_upper, min_area, roi_mask=None, exclude_mask=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)
    if exclude_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))
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

    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
