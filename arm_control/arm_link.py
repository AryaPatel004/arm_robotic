import glob
import logging
import queue
import threading
import time

import serial

log = logging.getLogger("arm_link")


class ArmLinkError(Exception):
    pass


class OverrideActiveError(ArmLinkError):
    pass


def resolve_port(port):
    if port != "auto":
        return port
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    for key in ("1a86", "CH340", "CH910", "CP210", "Silicon_Labs", "FTDI"):
        for p in by_id:
            if key.lower() in p.lower():
                return p
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise ArmLinkError("no ESP32 serial port found - check the USB cable and `ls /dev/ttyUSB* /dev/ttyACM*`")


def _parse_angles(reply):
    parts = reply.split(",")
    if len(parts) == 7:
        try:
            return [float(p) for p in parts[2:]]
        except ValueError:
            return None
    return None


class ArmLink:
    def __init__(self, port, baud=115200, timeout=0.2, slew_ms_per_deg=15.0):
        self.port = resolve_port(port)
        self.ser = serial.Serial(self.port, baud, timeout=timeout)
        self.override_active = False
        self.override_count = 0
        self.slew_s_per_deg = slew_ms_per_deg / 1000.0
        self.last_angles = None
        self._busy_until = 0.0
        self._replies = queue.Queue()
        self._lock = threading.Lock()
        self._stop = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        time.sleep(2.0)
        self._sync()
        log.info(f"ESP32 link up on {self.port}")

    def _read_loop(self):
        while not self._stop:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except (serial.SerialException, OSError):
                break
            if not line:
                continue
            if line.startswith("EVT,OVERRIDE_TRIGGERED"):
                self.override_active = True
                self.override_count += 1
                log.warning("kill switch pressed")
            elif line.startswith("EVT,OVERRIDE_CLEARED"):
                self.override_active = False
                log.info("kill switch released")
            elif line.startswith(("OK,", "ERR,")):
                self._replies.put(line)

    def _drain(self):
        while not self._replies.empty():
            try:
                self._replies.get_nowait()
            except queue.Empty:
                break

    def _sync(self, attempts=10):
        self.ser.reset_input_buffer()
        for _ in range(attempts):
            self._drain()
            self.ser.write(b"PING\n")
            deadline = time.time() + 0.5
            while True:
                left = deadline - time.time()
                if left <= 0:
                    break
                try:
                    if self._replies.get(timeout=left) == "OK,PING":
                        time.sleep(0.1)
                        self._drain()
                        return
                except queue.Empty:
                    break
        raise ArmLinkError(f"ESP32 on {self.port} not answering PING")

    def _send(self, cmd, timeout=2.0):
        with self._lock:
            self._drain()
            self.ser.write((cmd + "\n").encode())
            try:
                return self._replies.get(timeout=timeout)
            except queue.Empty:
                raise ArmLinkError(f"no reply to {cmd!r}")

    def _track(self, reply):
        angles = _parse_angles(reply)
        if angles is None:
            return
        if self.last_angles is None:
            delta = 180.0
        else:
            delta = max(abs(a - b) for a, b in zip(angles, self.last_angles))
        self.last_angles = angles
        self._busy_until = max(self._busy_until, time.time()) + delta * self.slew_s_per_deg

    def move(self, base, shoulder, elbow, wrist, gripper):
        reply = self._send(f"MOVE,{base:.0f},{shoulder:.0f},{elbow:.0f},{wrist:.0f},{gripper:.0f}")
        if reply == "ERR,MOVE,OVERRIDE_ACTIVE":
            self.override_active = True
            raise OverrideActiveError(reply)
        if reply.startswith("ERR"):
            raise ArmLinkError(reply)
        self._track(reply)
        return reply

    def home(self):
        reply = self._send("HOME")
        if reply == "ERR,HOME,OVERRIDE_ACTIVE":
            self.override_active = True
            raise OverrideActiveError(reply)
        if reply.startswith("ERR"):
            raise ArmLinkError(reply)
        self._track(reply)
        return reply

    def ping(self):
        return self._send("PING")

    def resume(self):
        reply = self._send("RESUME")
        if reply.startswith("ERR"):
            raise ArmLinkError(reply)
        self._busy_until = time.time()
        return reply

    def busy_s(self):
        return max(0.0, self._busy_until - time.time())

    def wait_settled(self, margin_s=0.15):
        left = self._busy_until - time.time()
        if left > 0:
            time.sleep(left)
        time.sleep(margin_s)

    def close(self):
        self._stop = True
        try:
            self.ser.close()
        except serial.SerialException:
            pass


class FakeArmLink:
    def __init__(self, slew_ms_per_deg=15.0, realtime=False):
        self.override_active = False
        self.override_count = 0
        self.last_angles = None
        self.realtime = realtime
        self.slew_s_per_deg = slew_ms_per_deg / 1000.0
        self.history = []
        self._busy_until = 0.0

    def _track(self, angles):
        delta = 180.0 if self.last_angles is None else max(abs(a - b) for a, b in zip(angles, self.last_angles))
        self.last_angles = angles
        self._busy_until = max(self._busy_until, time.time()) + delta * self.slew_s_per_deg

    def move(self, base, shoulder, elbow, wrist, gripper):
        angles = [round(base), round(shoulder), round(elbow), round(wrist), round(gripper)]
        self.history.append(("MOVE", angles))
        self._track(angles)
        log.info("DRY MOVE %s", angles)
        return "OK,MOVE," + ",".join(str(a) for a in angles)

    def home(self):
        self.history.append(("HOME", None))
        log.info("DRY HOME")
        return "OK,HOME"

    def ping(self):
        return "OK,PING"

    def resume(self):
        return "OK,RESUME"

    def busy_s(self):
        if not self.realtime:
            return 0.0
        return max(0.0, self._busy_until - time.time())

    def wait_settled(self, margin_s=0.15):
        if self.realtime:
            left = self._busy_until - time.time()
            if left > 0:
                time.sleep(left)
            time.sleep(margin_s)

    def close(self):
        pass
