import cv2
import numpy as np


class BackgroundSubtractionDetector:
    def __init__(self, reference_frame_path, min_area, arm_hsv_lower=None, arm_hsv_upper=None,
                 blur_ksize=21, diff_threshold=30):
        reference = cv2.imread(reference_frame_path)
        if reference is None:
            raise RuntimeError(f"reference frame not found at {reference_frame_path} - "
                               f"press 'Capture background' in vision_server.py first")
        self.blur_ksize = blur_ksize
        self.diff_threshold = diff_threshold
        self.reference_gray = self._gray(reference)
        self.min_area = min_area
        self.arm_hsv_lower = arm_hsv_lower
        self.arm_hsv_upper = arm_hsv_upper
        self.roi = None
        self.kernel = np.ones((7, 7), np.uint8)

    def _gray(self, frame):
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.blur_ksize, self.blur_ksize), 0)

    def set_roi(self, roi_mask):
        self.roi = roi_mask

    def mask(self, frame, exclude_mask=None):
        gray = self._gray(frame)
        if gray.shape != self.reference_gray.shape:
            raise RuntimeError(f"frame {gray.shape[::-1]} != reference {self.reference_gray.shape[::-1]} - "
                               f"recapture the background at the current resolution")
        diff = cv2.absdiff(gray, self.reference_gray)
        _, m = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        if self.roi is not None:
            m = cv2.bitwise_and(m, self.roi)
        if exclude_mask is not None:
            m = cv2.bitwise_and(m, cv2.bitwise_not(exclude_mask))
        if self.arm_hsv_lower is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            arm = cv2.inRange(hsv, np.array(self.arm_hsv_lower), np.array(self.arm_hsv_upper))
            m = cv2.bitwise_and(m, cv2.bitwise_not(arm))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)

    def detect_all(self, frame, exclude_mask=None):
        m = self.mask(frame, exclude_mask)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_area]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def detect(self, frame, exclude_mask=None):
        boxes = self.detect_all(frame, exclude_mask)
        return boxes[0] if boxes else None


class TFLiteObstacleDetector:
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
        self.roi = None

    def set_roi(self, roi_mask):
        self.roi = roi_mask

    def detect_all(self, frame, exclude_mask=None):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, self.input_size)
        input_data = np.expand_dims(resized, axis=0).astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        out = []
        for box, score in zip(boxes, scores):
            if score < self.score_threshold:
                continue
            ymin, xmin, ymax, xmax = box
            bx, by = int(xmin * w), int(ymin * h)
            bw, bh = int((xmax - xmin) * w), int((ymax - ymin) * h)
            cx, cy = min(w - 1, max(0, bx + bw // 2)), min(h - 1, max(0, by + bh // 2))
            if exclude_mask is not None and exclude_mask[cy, cx]:
                continue
            if self.roi is not None and not self.roi[cy, cx]:
                continue
            out.append((bx, by, bw, bh))
        return out

    def detect(self, frame, exclude_mask=None):
        boxes = self.detect_all(frame, exclude_mask)
        return boxes[0] if boxes else None


def build_obstacle_detector(cfg):
    if cfg.OBSTACLE_BACKEND == "tflite":
        return TFLiteObstacleDetector(cfg.TFLITE_MODEL_PATH)
    return BackgroundSubtractionDetector(
        cfg.REFERENCE_FRAME_PATH, cfg.OBSTACLE_MIN_AREA_PX, cfg.ARM_HSV_LOWER, cfg.ARM_HSV_UPPER,
        diff_threshold=cfg.OBSTACLE_DIFF_THRESHOLD,
    )
