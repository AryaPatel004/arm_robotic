import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2

import config as cfg
import vision_core as vc
from cam import open_source, default_source
from path_trace import find_path_waypoints

SRC = sys.argv[1] if len(sys.argv) > 1 else default_source()
PORT = 8000
MODES = ("raw", "aruco", "markers", "detect", "path")

lock = threading.Lock()
latest_jpg = None
latest_data = {}
cmds = []
state = {"mode": "raw", "target": "green", "msg": "starting"}


def draw_label(img, text, xy, col):
    cv2.putText(img, text, (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)


def worker():
    global latest_jpg, latest_data
    cap = None
    fails = 0
    mm_cfg = vc.load_markers_mm()
    ws = vc.load_workspace()
    bg, bg_acc = None, None
    click = None
    fps, t_prev = 0.0, time.time()
    while True:
        if cap is None:
            try:
                cap = open_source(SRC, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT, cfg.CAMERA_FPS)
                state["msg"] = f"camera open {SRC}"
            except SystemExit:
                state["msg"] = f"cannot open {SRC}"
                time.sleep(1)
                continue
        ok, f = cap.read()
        if not ok:
            fails += 1
            if fails > 30:
                cap.release()
                cap = None
                fails = 0
            time.sleep(0.05)
            continue
        fails = 0
        with lock:
            pending = cmds[:]
            cmds.clear()
        gray = vc.prep_gray(f)
        h, s, v = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))
        if ws is not None and ws["resolution"] != [f.shape[1], f.shape[0]]:
            ws = None
            state["msg"] = "workspace.json resolution mismatch - recalibrate"
        for c, arg in pending:
            if c == "mode" and arg in MODES:
                state["mode"] = arg
                state["msg"] = f"mode {arg}"
            elif c == "target" and arg in vc.COLORS:
                state["target"] = arg
                state["msg"] = f"target {arg}"
            elif c == "bg":
                bg_acc = []
                state["msg"] = "capturing background - keep workspace EMPTY"
            elif c == "calibrate":
                mm_cfg = vc.load_markers_mm()
                found, _, _ = vc.aruco_find(f)
                new = vc.save_calibration(found, mm_cfg, gray.shape)
                if new is not None:
                    ws = new
                    state["msg"] = "saved workspace.json + homography.json"
                else:
                    state["msg"] = f"calibration failed: see {sorted(found)}, need {sorted(mm_cfg)}"
            elif c == "snap":
                name = os.path.join(vc.HERE, f"snap_{int(time.time())}.png")
                cv2.imwrite(name, f)
                state["msg"] = f"saved {os.path.basename(name)}"
            elif c == "click" and arg:
                try:
                    x, y = (int(float(t)) for t in arg.split(","))
                except ValueError:
                    continue
                click = (x, y)
                hsv_px = [int(h[y, x]), int(s[y, x]), int(v[y, x])] if 0 <= y < h.shape[0] and 0 <= x < h.shape[1] else None
                if ws is not None:
                    mx, my = vc.to_mm(ws, [[x, y]])[0]
                    state["msg"] = f"click ({x},{y}) = ({mx:.1f}, {my:.1f}) mm  hsv {hsv_px}"
                else:
                    state["msg"] = f"click ({x},{y}) hsv {hsv_px} - not calibrated"
        if bg_acc is not None:
            bg_acc.append(gray.astype(np.float32))
            if len(bg_acc) >= 30:
                bg = np.mean(bg_acc, axis=0).astype(np.uint8)
                bg_acc = None
                cv2.imwrite(vc.REFERENCE_FRAME_PATH, f)
                state["msg"] = "background captured, saved reference_frame.png"
        roi = vc.roi_mask(ws, gray.shape)
        mode = state["mode"]
        data = {}
        if mode == "aruco":
            found, corners, ids = vc.aruco_find(f)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(f, corners, ids)
            data["aruco_seen"] = {str(i): [int(p[0]), int(p[1])] for i, p in found.items()}
            data["aruco_needed"] = sorted(mm_cfg)
        elif mode == "markers":
            items = vc.find_markers(h, s, v, ws, roi)
            for info, cnt in items:
                col = vc.DRAW[info["color"]]
                cv2.drawContours(f, [cnt], -1, col, 2)
                draw_label(f, f"{info['color']} {info['x']:.0f},{info['y']:.0f} {info['angle']:.0f}deg", info["px"], col)
            data["markers"] = [info for info, _ in items]
        elif mode == "detect":
            if bg is None:
                if bg_acc is None:
                    state["msg"] = "press Capture background (workspace empty)"
            else:
                pen, pen_c, obs, hulls = vc.detect(h, s, v, gray, bg, ws, roi, state["target"])
                if pen_c is not None and pen is not None:
                    cv2.drawContours(f, [pen_c], -1, (0, 255, 0), 2)
                    draw_label(f, f"PEN {pen['x']:.0f},{pen['y']:.0f} {pen['angle']:.0f}deg", pen["px"], (0, 255, 0))
                cv2.drawContours(f, hulls, -1, (0, 0, 255), 2)
                data["pen"] = pen
                data["obstacles"] = obs
        elif mode == "path":
            pts = find_path_waypoints(f, cfg.PATH_HSV_LOWER, cfg.PATH_HSV_UPPER,
                                      min_length_px=cfg.PATH_MIN_LENGTH_PX, sample_every=cfg.PATH_SAMPLE_EVERY_PX,
                                      roi_mask=roi if ws is not None else None)
            if pts:
                arr = np.array(pts, np.int32)
                cv2.polylines(f, [arr], False, (255, 0, 255), 2)
                cv2.circle(f, tuple(int(c) for c in arr[0]), 7, (0, 255, 0), -1)
                cv2.circle(f, tuple(int(c) for c in arr[-1]), 7, (0, 0, 255), -1)
                data["path_px"] = [[int(a), int(b)] for a, b in pts]
                if ws is not None:
                    data["path_mm"] = [[round(float(a), 1), round(float(b), 1)] for a, b in vc.to_mm(ws, pts)]
            else:
                data["path_px"] = None
        if ws is not None:
            cv2.polylines(f, [ws["roi"]], True, (255, 255, 0), 1)
        if click is not None:
            cv2.drawMarker(f, click, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        units = "mm" if ws is not None else "px"
        cv2.putText(f, f"{mode} | {fps:.1f} fps | {units}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(f, state["msg"][:70], (10, f.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        data.update({
            "t": round(now, 3), "mode": mode, "target": state["target"], "fps": round(fps, 1),
            "units": units, "calibrated": ws is not None, "background": bg is not None, "msg": state["msg"],
        })
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            if ok:
                latest_jpg = buf.tobytes()
            latest_data = data


OPTIONS = "".join(f"<option{' selected' if c == 'green' else ''}>{c}</option>" for c in vc.COLORS)
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UNO Q Vision</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
button,select{margin:4px 4px 4px 0;padding:8px 12px;font-size:14px}
img{display:block;max-width:100%;margin:12px 0;border:1px solid #444;cursor:crosshair}
pre{background:#1d1d1d;padding:10px;max-height:320px;overflow:auto;font-size:12px}
</style></head><body>
<h3>UNO Q Vision</h3>
<div>Mode:
<button onclick="cmd('mode','raw')">Raw</button>
<button onclick="cmd('mode','aruco')">ArUco</button>
<button onclick="cmd('mode','markers')">Markers</button>
<button onclick="cmd('mode','detect')">Detect</button>
<button onclick="cmd('mode','path')">Path</button></div>
<div>Target pen: <select onchange="cmd('target',this.value)">OPTIONS</select>
<button onclick="cmd('bg')">Capture background</button>
<button onclick="cmd('calibrate')">Save calibration</button>
<button onclick="cmd('snap')">Snapshot</button></div>
<img id="v" src="/stream">
<pre id="d">waiting...</pre>
<script>
function cmd(c,v){fetch('/cmd?c='+c+(v!==undefined?'&v='+encodeURIComponent(v):''))}
document.getElementById('v').addEventListener('click',e=>{const i=e.target;cmd('click',Math.round(e.offsetX*i.naturalWidth/i.clientWidth)+','+Math.round(e.offsetY*i.naturalHeight/i.clientHeight))})
setInterval(()=>fetch('/data').then(r=>r.json()).then(j=>{document.getElementById('d').textContent=JSON.stringify(j,null,1)}).catch(()=>{}),500)
</script></body></html>
""".replace("OPTIONS", OPTIONS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self.send_body(PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/data":
            with lock:
                body = json.dumps(latest_data).encode()
            self.send_body(body, "application/json")
        elif u.path == "/cmd":
            with lock:
                cmds.append((q.get("c", [""])[0], q.get("v", [None])[0]))
            self.send_body(b"ok", "text/plain")
        elif u.path == "/stream":
            self.stream()
        else:
            self.send_error(404)

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while True:
                with lock:
                    jpg = latest_jpg
                if jpg is None or jpg is last:
                    time.sleep(0.01)
                    continue
                last = jpg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    print(f"camera {SRC}")
    print(f"open http://<UNO-Q-IP>:{PORT} in your laptop browser")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
