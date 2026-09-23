"""Obstacle detection, behind one interface with two swappable backends:

  - BackgroundSubtractionDetector: classical CV, works today with zero training data.
  - TFLiteObstacleDetector: runs a trained model (e.g. exported from Edge Impulse) once
    one actually exists.

Both expose detect(frame) -> (u, v, w, h) pixel bounding box, or None. Swap OBSTACLE_BACKEND
in config.py to move from one to the other without touching planner.py or state_machine.py.
"""

import cv2
import numpy as np


class BackgroundSubtractionDetector:
    """Flags any new blob against a stored empty-workspace reference frame. Known
    limitation: if the arm's own body/gripper isn't excluded via ARM_HSV_LOWER/UPPER,
    it can occasionally register as its own obstacle - tune min_area and the arm color
    exclusion before trusting this near the arm's resting position."""

    def __init__(self, reference_frame_path, min_area, arm_hsv_lower=None, arm_hsv_upper=None, blur_ksize=21):
        reference = cv2.imread(reference_frame_path)
        if reference is None:
            raise RuntimeError(f"reference frame not found at {reference_frame_path} - run calibrate.py first")
        self.blur_ksize = blur_ksize
        self.reference_gray = cv2.GaussianBlur(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (blur_ksize, blur_ksize), 0
        )
        self.min_area = min_area
        self.arm_hsv_lower = arm_hsv_lower
        self.arm_hsv_upper = arm_hsv_upper

    def detect(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.blur_ksize, self.blur_ksize), 0)
        diff = cv2.absdiff(gray, self.reference_gray)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        if self.arm_hsv_lower is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            arm_mask = cv2.inRange(hsv, np.array(self.arm_hsv_lower), np.array(self.arm_hsv_upper))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(arm_mask))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None
        return cv2.boundingRect(largest)


class TFLiteObstacleDetector:
    """Wraps a trained TFLite model behind the same interface. Assumes the standard
    TFLite object-detection output layout (boxes, classes, scores, count) - adjust
    detect() if your Edge Impulse export differs."""

    def __init__(self, model_path, input_size=(96, 96), score_threshold=0.6):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.score_threshold = score_threshold

    def detect(self, frame):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, self.input_size)
        input_data = np.expand_dims(resized, axis=0).astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        best = None
        for box, score in zip(boxes, scores):
            if score < self.score_threshold:
                continue
            if best is None or score > best[1]:
                best = (box, score)
        if best is None:
            return None

        ymin, xmin, ymax, xmax = best[0]
        return (int(xmin * w), int(ymin * h), int((xmax - xmin) * w), int((ymax - ymin) * h))


def build_obstacle_detector(cfg):
    if cfg.OBSTACLE_BACKEND == "tflite":
        return TFLiteObstacleDetector(cfg.TFLITE_MODEL_PATH)
    return BackgroundSubtractionDetector(
        cfg.REFERENCE_FRAME_PATH, cfg.OBSTACLE_MIN_AREA_PX, cfg.ARM_HSV_LOWER, cfg.ARM_HSV_UPPER
    )
