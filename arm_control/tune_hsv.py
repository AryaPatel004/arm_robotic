"""Interactive HSV threshold tuner. Adjust the trackbars until only your target (the
marker, or the drawn path) shows white in the mask window, then copy the printed values
into config.py's MARKER_HSV_* or PATH_HSV_* constants. Run this under the actual venue
lighting, not wherever you first tested - lighting shifts HSV ranges more than you'd expect.
"""

import cv2

import config
from camera import Camera


def _nothing(_):
    pass


def main():
    cv2.namedWindow("tune")
    cv2.createTrackbar("H min", "tune", 0, 180, _nothing)
    cv2.createTrackbar("S min", "tune", 0, 255, _nothing)
    cv2.createTrackbar("V min", "tune", 0, 255, _nothing)
    cv2.createTrackbar("H max", "tune", 180, 180, _nothing)
    cv2.createTrackbar("S max", "tune", 255, 255, _nothing)
    cv2.createTrackbar("V max", "tune", 255, 255, _nothing)

    with Camera(config.CAMERA_DEVICE, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS) as cam:
        while True:
            frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = (
                cv2.getTrackbarPos("H min", "tune"),
                cv2.getTrackbarPos("S min", "tune"),
                cv2.getTrackbarPos("V min", "tune"),
            )
            upper = (
                cv2.getTrackbarPos("H max", "tune"),
                cv2.getTrackbarPos("S max", "tune"),
                cv2.getTrackbarPos("V max", "tune"),
            )
            mask = cv2.inRange(hsv, lower, upper)
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"lower={lower}, upper={upper}")
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
