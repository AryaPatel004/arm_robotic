"""Implements exactly the serial protocol from esp32_dev_brief.md: MOVE / HOME / PING /
RESUME requests, and asynchronous EVT,OVERRIDE_TRIGGERED / EVT,OVERRIDE_CLEARED events
that can arrive at any time, independent of whatever command is in flight.
"""

import queue
import threading
import time

import serial


class ArmLinkError(Exception):
    pass


class ArmLink:
    def __init__(self, port, baud=115200, timeout=2.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.override_active = False
        self._replies = queue.Queue()
        self._lock = threading.Lock()
        self._stop = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        time.sleep(2.0)  # the ESP32 resets when the serial port opens - give it time to boot

    def _read_loop(self):
        while not self._stop:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except serial.SerialException:
                break
            if not line:
                continue
            if line.startswith("EVT,OVERRIDE_TRIGGERED"):
                self.override_active = True
            elif line.startswith("EVT,OVERRIDE_CLEARED"):
                self.override_active = False
            else:
                self._replies.put(line)

    def _send(self, cmd, timeout=3.0):
        with self._lock:
            self.ser.write((cmd + "\n").encode())
            try:
                return self._replies.get(timeout=timeout)
            except queue.Empty:
                raise ArmLinkError(f"no reply to {cmd!r}")

    def move(self, base, shoulder, elbow, gripper):
        reply = self._send(f"MOVE,{base:.0f},{shoulder:.0f},{elbow:.0f},{gripper:.0f}")
        if reply.startswith("ERR"):
            raise ArmLinkError(reply)
        return reply

    def home(self):
        reply = self._send("HOME")
        if reply.startswith("ERR"):
            raise ArmLinkError(reply)
        return reply

    def ping(self):
        return self._send("PING")

    def resume(self):
        reply = self._send("RESUME")
        if reply.startswith("ERR"):
            raise ArmLinkError(reply)
        return reply

    def close(self):
        self._stop = True
        self.ser.close()
