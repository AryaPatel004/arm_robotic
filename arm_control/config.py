"""Central configuration.

Anything marked MEASURE ON BENCH is a placeholder, not a verified value - same convention
as esp32_dev_brief.md. Do not trust these numbers until you've checked them against the
real hardware; they're here so every other module has one place to import constants from.
"""

import os

# --- Camera ------------------------------------------------------------------
CAMERA_DEVICE = 0  # MEASURE ON BENCH: once the C270 is on the UNO Q, check
                    # `v4l2-ctl --list-devices` - it may not land on /dev/video0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# --- Serial link to the ESP32 (must match esp32_dev_brief.md's protocol) -----
SERIAL_PORT = "/dev/ttyUSB0"  # MEASURE ON BENCH: `ls /dev/ttyUSB* /dev/ttyACM*`
SERIAL_BAUD = 115200

# --- Calibration artifacts (written by calibrate.py) --------------------------
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
HOMOGRAPHY_PATH = os.path.join(CONFIG_DIR, "homography.json")
REFERENCE_FRAME_PATH = os.path.join(CONFIG_DIR, "reference_frame.png")

# --- Marker (pen) detection - classical HSV threshold --------------------------
# MEASURE ON BENCH: placeholder range for a generic orange cap. Use tune_hsv.py
# against the real marker under real venue lighting before trusting this.
MARKER_HSV_LOWER = (5, 150, 100)
MARKER_HSV_UPPER = (18, 255, 255)
MARKER_MIN_AREA_PX = 200

# --- Path (drawn trace) detection -----------------------------------------------
# MEASURE ON BENCH: placeholder for dark ink on light paper.
PATH_HSV_LOWER = (0, 0, 0)
PATH_HSV_UPPER = (180, 255, 60)

# --- Obstacle detection -----------------------------------------------------------
OBSTACLE_BACKEND = "cv"  # "cv" = background subtraction, works today with no training.
                          # "tflite" = a trained model (e.g. exported from Edge Impulse) -
                          # swap to this once obstacle_model.tflite actually exists.
OBSTACLE_MIN_AREA_PX = 1500
TFLITE_MODEL_PATH = os.path.join(CONFIG_DIR, "obstacle_model.tflite")
# MEASURE ON BENCH: HSV range of the arm's own material/paint, to stop the arm's
# own body from ever being flagged as an obstacle against the background reference.
# Set both to None to disable this exclusion.
ARM_HSV_LOWER = None
ARM_HSV_UPPER = None

# --- Arm geometry (mm) ----------------------------------------------------------------
# Measured directly off the physical arm (labeled on the links, see IMG_4283). Confirmed:
# only 3 positioning servos exist (base/shoulder/elbow) - no wrist joint. See
# kinematics.py's docstring re: whether pen orientation is passively self-leveling
# (parallel four-bar links) or drifts with reach (simple rigid links) - still unconfirmed.
# Servos are ACEBOTT MG90S.
LINK1_MM = 78.0         # shoulder pivot -> elbow pivot
LINK2_MM = 50.0         # elbow pivot -> gripper mount / pen-tip
BASE_HEIGHT_MM = 40.0   # table surface -> shoulder pivot height

# --- ESP32 GPIO map (reference only - Python never touches these directly; the ESP32
# firmware does. Kept here as the single source of truth alongside esp32_dev_brief.md) --
# S1 base     -> GPIO5      S2 shoulder -> GPIO16
# S3 elbow    -> GPIO17     S4 gripper  -> GPIO18
# J1 (left)   X -> GPIO32   Y -> GPIO33   SW -> GPIO34
# J2 (right)  X -> GPIO35   Y -> GPIO36   SW -> GPIO39
# Note: GPIO34/35/36/39 are input-only with no internal pull-up - the J1/J2 SW button
# reads depend on the joystick module supplying its own pull-up, or they'll float.

PEN_UP_Z_MM = 40.0      # travel height above the paper
PEN_DOWN_Z_MM = 0.0     # pen touching the paper - your homography's world-Z=0 plane

HOME_XY_MM = (0.0, 120.0)   # MEASURE ON BENCH: a safe, not-overextended resting XY
GRIPPER_OPEN_DEG = 30.0     # MEASURE ON BENCH
GRIPPER_CLOSED_DEG = 90.0   # MEASURE ON BENCH

JOINT_LIMITS_DEG = {  # MEASURE ON BENCH - mirror whatever limits you set in the ESP32 firmware
    "base": (0, 180),
    "shoulder": (0, 180),
    "elbow": (0, 180),
    "wrist": (0, 180),
}

# --- Guarded move / replanner --------------------------------------------------------
STEP_MM = 5.0             # distance covered per guarded-move increment
DETOUR_CLEARANCE_MM = 30.0
MAX_STEPS_PER_MOVE = 500
