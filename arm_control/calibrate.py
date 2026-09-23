"""One-time interactive setup: click 4 known points on the work surface to build the
pixel<->world homography, and capture the empty-workspace reference frame the classical
obstacle detector diffs against. Run this once per camera mount position - if the camera
or table moves, re-run it.
"""

import json
import sys

import cv2

import config
from camera import Camera


def main():
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"captured pixel point {len(points)}: ({x},{y})")

    with Camera(config.CAMERA_DEVICE, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS) as cam:
        cv2.namedWindow("calibrate")
        cv2.setMouseCallback("calibrate", on_click)
        print("Click 4 known points on the work surface, in order.")
        print("Press 'r' at any time to save the current frame as the empty-workspace reference.")
        print("Press 'q' once you have 4 points and a reference frame.")

        last_frame = None
        while True:
            frame = cam.read()
            last_frame = frame.copy()
            for i, p in enumerate(points):
                cv2.circle(frame, p, 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("calibrate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                cv2.imwrite(config.REFERENCE_FRAME_PATH, last_frame)
                print(f"saved reference frame to {config.REFERENCE_FRAME_PATH}")
            if key == ord("q"):
                break
    cv2.destroyAllWindows()

    if len(points) != 4:
        sys.exit(f"need exactly 4 points, got {len(points)} - aborting without writing homography.json")

    world_points = []
    print("\nNow enter the real-world XY (mm) for each point, same order, measured from")
    print("whatever origin you want the arm's coordinates to use (e.g. the arm's base).")
    for i in range(4):
        x = float(input(f"point {i + 1} X (mm): "))
        y = float(input(f"point {i + 1} Y (mm): "))
        world_points.append([x, y])

    with open(config.HOMOGRAPHY_PATH, "w") as f:
        json.dump({"pixel_points": points, "world_points_mm": world_points}, f, indent=2)
    print(f"saved homography to {config.HOMOGRAPHY_PATH}")


if __name__ == "__main__":
    main()
