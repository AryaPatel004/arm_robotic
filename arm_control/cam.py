import sys
import glob
import cv2


def default_source():
    if not sys.platform.startswith("linux"):
        return "0"
    hits = sorted(glob.glob("/dev/v4l/by-id/*C270*-video-index0")) or sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    if not hits:
        raise SystemExit("no USB camera found - check lsusb and the powered hub")
    return hits[0]


def is_live(src):
    return src.isdigit() or src.startswith("/dev/")


def open_source(src="0", width=640, height=480, fps=30):
    if is_live(src):
        if sys.platform.startswith("linux"):
            api = cv2.CAP_V4L2
        elif sys.platform == "win32":
            api = cv2.CAP_DSHOW
        else:
            api = cv2.CAP_ANY
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src, api)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    return cap
