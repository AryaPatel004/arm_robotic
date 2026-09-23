#!/usr/bin/env bash
DEV=${1:-$(ls /dev/v4l/by-id/*C270*-video-index0 2>/dev/null | head -n1)}
if [ -z "$DEV" ]; then echo "C270 not found"; exit 1; fi
echo "device $DEV"
set_ctrl() { v4l2-ctl -d "$DEV" -c "$1" 2>/dev/null && echo "set $1"; }
set_ctrl power_line_frequency=2
set_ctrl white_balance_automatic=0 || set_ctrl white_balance_temperature_auto=0
set_ctrl white_balance_temperature=${WB:-4600}
set_ctrl auto_exposure=1 || set_ctrl exposure_auto=1
set_ctrl exposure_time_absolute=${EXPOSURE:-250} || set_ctrl exposure_absolute=${EXPOSURE:-250}
set_ctrl gain=0
set_ctrl backlight_compensation=0
set_ctrl exposure_dynamic_framerate=0
v4l2-ctl -d "$DEV" --list-ctrls
