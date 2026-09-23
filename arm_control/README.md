# arm_control

Runs on the Arduino UNO Q (Debian side). One Python process, no ROS2.

| File | Role |
|---|---|
| `config.py` | Every tunable constant |
| `camera.py` | C270 capture, always returns a fresh frame |
| `vision_server.py` | Browser tool on port 8000: calibration, background capture, HSV readout, path preview |
| `vision_core.py`, `cam.py` | Helpers for `vision_server.py` |
| `probe.py` | Camera check |
| `marker.py` | Find the pen by color |
| `path_trace.py` | Ordered waypoints from the drawn line |
| `obstacle.py` | Obstacle detection (background subtraction, or a TFLite model) |
| `homography.py` | Pixel <-> mm, plus world-space masks |
| `kinematics.py` | IK and IK -> servo angle mapping |
| `arm_link.py` | Serial protocol to the ESP32, plus a fake link for dry runs |
| `planner.py` | Guarded move with obstacle detours |
| `state_machine.py` | Home -> find pen -> find path -> grasp -> trace -> place -> home |
| `main.py` | Entry point |

## Run order

1. `./setup.sh` (once)
2. `python probe.py`
3. `python vision_server.py` then in the browser: ArUco -> Save calibration, Capture background, Markers (read HSV), Path (check the line)
4. `python main.py --dry-run`
5. `python main.py`
