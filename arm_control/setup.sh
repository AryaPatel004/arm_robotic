#!/usr/bin/env bash
set -e
sudo apt update
sudo apt install -y python3-venv python3-pip v4l-utils
sudo usermod -aG dialout,video "$USER"
python3 -m venv ~/pandav/.venv
. ~/pandav/.venv/bin/activate
pip install --upgrade pip
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >/dev/null 2>&1 || true
pip install -r "$(dirname "$0")/requirements.txt"
python -c "import cv2, serial; cv2.aruco.ArucoDetector; print('opencv', cv2.__version__, 'pyserial', serial.VERSION, 'OK')"
echo "log out and back in once so the dialout group applies"
