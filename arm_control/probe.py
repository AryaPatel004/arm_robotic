import sys
import time

import cv2

import config as cfg
from camera import Camera

src = sys.argv[1] if len(sys.argv) > 1 else cfg.CAMERA_DEVICE
with Camera(src, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT, cfg.CAMERA_FPS) as cam:
    fourcc = int(cam.cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    frame = cam.read()
    print(f"res {frame.shape[1]}x{frame.shape[0]}  fourcc {fourcc}")
    t0 = time.time()
    for _ in range(90):
        frame = cam.read()
    dt = time.time() - t0
    print(f"90 fresh frames in {dt:.1f}s = {90 / dt:.1f} fps")
    cv2.imwrite("probe.jpg", frame)
    print("saved probe.jpg")
