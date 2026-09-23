import argparse
import logging

import config as cfg
from arm_link import ArmLink, FakeArmLink
from camera import Camera
from homography import Homography
from kinematics import make_ik_solver
from obstacle import build_obstacle_detector
from state_machine import ArmController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=cfg.CAMERA_DEVICE)
    parser.add_argument("--port", default=cfg.SERIAL_PORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", default="run_log.csv")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file)],
    )

    camera = Camera(args.camera, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT, cfg.CAMERA_FPS)
    homography = Homography(cfg.HOMOGRAPHY_PATH)
    if args.dry_run:
        arm_link = FakeArmLink(cfg.SLEW_MS_PER_DEG, realtime=True)
    else:
        arm_link = ArmLink(args.port, cfg.SERIAL_BAUD, slew_ms_per_deg=cfg.SLEW_MS_PER_DEG)
    obstacle_detector = build_obstacle_detector(cfg)
    controller = ArmController(camera, homography, arm_link, obstacle_detector, make_ik_solver(cfg))

    try:
        controller.run_once()
    except KeyboardInterrupt:
        logging.getLogger("main").warning("stopped by user")
    finally:
        arm_link.close()
        camera.close()


if __name__ == "__main__":
    main()
