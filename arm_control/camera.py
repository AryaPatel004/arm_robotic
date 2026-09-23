import glob
import sys
import threading
import time

import cv2


def resolve_device(device):
    if isinstance(device, int) or str(device).isdigit():
        return int(device)
    if device == "auto":
        if not sys.platform.startswith("linux"):
            return 0
        hits = sorted(glob.glob("/dev/v4l/by-id/*C270*-video-index0")) or sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
        if not hits:
            raise RuntimeError("no USB camera found - check lsusb and the powered hub")
        return hits[0]
    return device


class Camera:
    def __init__(self, device, width, height, fps):
        dev = resolve_device(device)
        self.is_file = isinstance(dev, str) and not dev.startswith("/dev/")
        self.period = 1.0 / fps
        if self.is_file:
            api = cv2.CAP_ANY
        elif sys.platform.startswith("linux"):
            api = cv2.CAP_V4L2
        elif sys.platform == "win32":
            api = cv2.CAP_DSHOW
        else:
            api = cv2.CAP_ANY
        self.cap = cv2.VideoCapture(dev, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {dev}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._seq = 0
        self._error = None
        self._running = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.read(timeout=5.0)

    def _loop(self):
        fails = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok and self.is_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if ok and self.is_file:
                time.sleep(self.period)
            if not ok:
                fails += 1
                if fails > 50:
                    with self._cond:
                        self._error = "camera read failed"
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            fails = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout=2.0):
        with self._cond:
            seq = self._seq
            if not self._cond.wait_for(lambda: self._seq > seq or self._error or not self._running, timeout):
                raise RuntimeError("camera read timed out")
            if self._error:
                raise RuntimeError(self._error)
            if self._frame is None:
                raise RuntimeError("camera closed")
            return self._frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
