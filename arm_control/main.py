"""Entry point - run this on the UNO Q once calibrate.py has produced homography.json
and reference_frame.png, and the ESP32 firmware is flashed and bench-tested standalone.
"""

import argparse
import logging

import config as cfg
from arm_link import ArmLink
from camera import Camera
from homography import Homography
from kinematics import solve_ik
from obstacle import build_obstacle_detector
from state_machine import ArmController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=cfg.CAMERA_DEVICE)
    parser.add_argument("--port", default=cfg.SERIAL_PORT)
    parser.add_argument("--log-file", default="run_log.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file)],
    )

    camera = Camera(args.camera, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT, cfg.CAMERA_FPS)
    homography = Homography(cfg.HOMOGRAPHY_PATH)
    arm_link = ArmLink(args.port, cfg.SERIAL_BAUD)
    obstacle_detector = build_obstacle_detector(cfg)

    def ik_solver(x, y, z):
        return solve_ik(x, y, z, cfg.LINK1_MM, cfg.LINK2_MM, cfg.BASE_HEIGHT_MM)

    controller = ArmController(camera, homography, arm_link, obstacle_detector, ik_solver)

    try:
        controller.run_once()
    finally:
        arm_link.close()
        camera.close()


if __name__ == "__main__":
    main()
