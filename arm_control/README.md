# arm_control

Plain-Python perception + planning + actuation stack for the UNO Q side. No ROS2 — one
process, organized into the same modules we designed, each independently bench-testable.

## Files

| File | Role |
|---|---|
| `config.py` | Every tunable constant. Anything marked `MEASURE ON BENCH` is a placeholder. |
| `camera.py` | C270 capture wrapper. |
| `calibrate.py` | One-time: click 4 points → `homography.json`; capture `reference_frame.png`. |
| `tune_hsv.py` | Interactive trackbar tool to find the marker/path HSV ranges. |
| `marker.py` | Classical CV: locate the marker/pen by color. |
| `path_trace.py` | Extract ordered waypoints from the drawn path (needs `opencv-contrib-python`). |
| `obstacle.py` | Obstacle detection — classical background-subtraction today, or a trained TFLite model once one exists (same interface, swap via `config.OBSTACLE_BACKEND`). |
| `kinematics.py` | Analytical IK for the 4DOF arm. |
| `arm_link.py` | Serial link to the ESP32 — implements the exact protocol from `esp32_dev_brief.md`. |
| `planner.py` | The shared "Guarded Move" loop: step, check obstacle, replan detour if needed. |
| `state_machine.py` | The full flowchart: search → grasp → trace → place → home. |
| `main.py` | Wires it all together and runs one full pass. |

## Install (on the UNO Q's Debian side)

```bash
pip install -r requirements.txt
```

## Bring-up order — bench-test each piece before the next, same as the dev plan

1. **Camera**: `python3 -c "from camera import Camera; import config; c=Camera(config.CAMERA_DEVICE, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS); print(c.read().shape)"` — confirms the C270 opens and returns real frames before anything else.
2. **Calibration**: `python3 calibrate.py` — do this with the camera in its final mounted position; re-run if it moves.
3. **HSV tuning**: `python3 tune_hsv.py`, once for the marker's color, once for the path's ink color, under the actual venue lighting. Paste the printed values into `config.py`.
4. **Marker detection alone**: run `find_marker` against a live frame in a loop, print the result, move the pen by hand — confirm it tracks.
5. **Path detection alone**: same idea with `find_path_waypoints` against your drawn path.
6. **Obstacle detection alone**: wave a hand into frame, confirm `BackgroundSubtractionDetector.detect()` fires and clears correctly. Tune `OBSTACLE_MIN_AREA_PX` and `ARM_HSV_LOWER/UPPER` here if the arm's own body triggers false positives.
7. **ESP32 link**: confirm `esp32_dev_brief.md`'s firmware passes its own bench tests first (plain serial terminal), *then* test `ArmLink` from Python against it — `ping()`, a `move()`, trigger the kill switch mid-move and confirm `override_active` flips.
8. **Kinematics**: sanity-check `solve_ik` against a few known points by hand before trusting it near the arm's physical limits.
9. **Full run**: `python3 main.py`, only after all of the above pass individually.

## Known limitations, by design (v1, not oversights)

- `BackgroundSubtractionDetector` can mistake the arm's own body for an obstacle if `ARM_HSV_LOWER/UPPER` isn't tuned — the interface is deliberately identical to `TFLiteObstacleDetector` so this is a drop-in upgrade once you have a trained model, not a rewrite.
- `path_trace.py` assumes one open curve with no branches — matches "a path from start to finish," not a general drawing.
- IK raises `ValueError` on an out-of-reach target rather than clamping — keep your drawn path inside the arm's reach envelope; a silent clamp would just draw the wrong thing.
- Every position sent to the ESP32 is recomputed live each cycle — nothing here is a stored waypoint list, which is what the challenge asks for.
