#!/usr/bin/env python3
"""The Pi's Bluetooth link to the phone, as its own service (virtual-car-sound-ble).

The same Bluetooth LE service the engine used to run itself (ble_control.py: COMMAND, STATUS,
LIST, TELEMETRY), relayed from the engine over local sockets (ControlBridge), plus LOG: the
engine's log, streamed from the journal. Because it runs on its own, the phone stays connected
while the engine restarts, updates or is stuck, and can see what it's doing:

  - the engine's telemetry and status are passed straight on; the phone's commands go back
  - while the engine isn't reporting, the status says what systemd says it's doing
    ("svc": activating -- starting or updating --, failed, inactive...)
  - LOG notifies the journal of both services a line at a time ("HH:MM:SS message"); the
    command "logs" replays the recent ones (the app asks when it connects)
  - when its own code changes (an update), it restarts itself once no phone is connected

    python ble_service.py          (installed by install_service.sh as virtual-car-sound-ble)
"""
from __future__ import annotations

import collections
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import ble_control as bc

HERE = Path(__file__).resolve().parent
ENGINE_UNIT = "virtual-car-sound"
BLE_UNIT = "virtual-car-sound-ble"
STATUS_FRESH_SEC = 3.0          # the engine's status counts as live this long
LOG_BACKLOG = 300               # journal lines kept for a phone that connects later
LOG_REPLAY = 120                # ...and how many "logs" sends
LOG_LINE_MAX = 180              # bytes per notification (fits any MTU the app negotiates)
LOG_PER_TICK = 4                # journal lines sent per 4 ms tick, at most
RELOAD_CHECK_SEC = 30.0
SVC_MESSAGES = {
    "activating": "The engine is starting (or updating). See the Pi's log.",
    "deactivating": "The engine is stopping.",
    "inactive": "The engine isn't running.",
    "failed": "The engine has stopped after a failure. See the Pi's log.",
}


def log(msg: str):
    print(msg, flush=True)


class BleService(bc.ControlServer):
    def __init__(self):
        super().__init__(engine=None, voice_names=[], save=lambda s: None, log=log)
        self._in = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            os.unlink(bc.BLE_SOCKET)
        except FileNotFoundError:
            pass
        self._in.bind(bc.BLE_SOCKET)
        self._in.setblocking(False)
        self._out = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.status: dict = {}
        self.status_at = 0.0
        self.telemetry = bytes([1, 0]) + bytes(8)      # version 1, not fresh, zeros
        self.message = ""
        self.backlog: collections.deque[str] = collections.deque(maxlen=LOG_BACKLOG)
        self.pending_log: collections.deque[bytes] = collections.deque(maxlen=LOG_BACKLOG)
        self._svc = "unknown"
        self._svc_at = 0.0
        self._sources = {p: p.stat().st_mtime for p in (HERE / "ble_service.py", HERE / "ble_control.py")}
        self._reload_at = time.monotonic()
        self.reload = False

    # --- the engine's side ----------------------------------------------------------------
    def _pump(self, now: float):
        while True:
            try:
                msg = self._in.recv(4096)
            except (BlockingIOError, OSError):
                break
            kind, data = msg[:1], msg[1:]
            if kind == b"T":
                self.telemetry = data
                self._notify(bc.TELEMETRY, data)
            elif kind == b"S":
                try:
                    self.status = json.loads(data)
                    self.status_at = time.monotonic()
                    self.message = ""
                except ValueError:
                    continue
                self._notify(bc.STATUS, self._status())
            elif kind == b"V":
                try:
                    self.voices = list(json.loads(data))
                except ValueError:
                    pass
        for _ in range(LOG_PER_TICK):                   # the journal, a few lines a tick
            if not self.pending_log:
                break
            self._notify(bc.LOG, self.pending_log.popleft())
        if time.monotonic() - self._reload_at >= RELOAD_CHECK_SEC:
            self._reload_at = time.monotonic()
            self._check_reload()

    def _engine_live(self) -> bool:
        return time.monotonic() - self.status_at < STATUS_FRESH_SEC

    def _engine_state(self) -> str:
        """What systemd says the engine is doing (asked at most every 2 s)."""
        if time.monotonic() - self._svc_at >= 2.0:
            self._svc_at = time.monotonic()
            try:
                self._svc = subprocess.run(["systemctl", "is-active", ENGINE_UNIT], capture_output=True,
                                           text=True, timeout=2).stdout.strip() or "unknown"
            except (OSError, subprocess.TimeoutExpired):
                self._svc = "unknown"
        return self._svc

    def _status(self) -> bytes:
        if self._engine_live():
            st = dict(self.status)
            st["svc"] = "running"
        else:                                           # the engine isn't reporting: say why
            svc = self._engine_state()
            st = {"v": self.status.get("v", ""), "vol": self.status.get("vol", 1.0), "car": False,
                  "cq": self.status.get("cq", True), "svc": svc,
                  "msg": self.message or SVC_MESSAGES.get(svc, "The engine isn't reporting.")}
        if self.message:
            st["msg"] = self.message
        return json.dumps(st, separators=(",", ":")).encode("utf-8")

    def _telemetry(self) -> bytes:
        return self.telemetry if self._engine_live() else bytes([1, 0]) + bytes(8)

    # --- the phone's side -----------------------------------------------------------------
    def _command(self, text: str):
        if text == "logs":                              # the recent log, for a phone that just connected
            lines = list(self.backlog)[-LOG_REPLAY:]
            self.pending_log.extendleft(reversed([ln.encode("utf-8")[:LOG_LINE_MAX] for ln in lines]))
            return
        if text == "voices":
            self.loop.create_task(self._send_list())
        try:
            self._out.sendto(text.encode("utf-8"), bc.MAIN_SOCKET)
            self.message = ""
        except OSError:
            if text not in ("voices", "status"):
                self.message = "The engine isn't running right now, so that didn't go through."
        self._notify(bc.STATUS, self._status())

    # --- the journal ----------------------------------------------------------------------
    def follow_journal(self):
        cmd = ["journalctl", "-u", ENGINE_UNIT, "-u", BLE_UNIT, "-f", "-n", str(LOG_REPLAY),
               "-o", "short-iso", "--no-hostname"]
        while not self._stop.is_set():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except OSError as e:
                self._add_log(f"can't read the journal: {e}")
                return
            for raw in proc.stdout:
                self._add_log(_tidy(raw))
            proc.wait()
            if self._stop.wait(5.0):
                return

    def _add_log(self, line: str):
        if not line:
            return
        self.backlog.append(line)
        self.pending_log.append(line.encode("utf-8")[:LOG_LINE_MAX])

    # --- updates --------------------------------------------------------------------------
    def _check_reload(self):
        """Our own code changed (an update): restart, once no phone would notice."""
        try:
            changed = any(p.stat().st_mtime != m for p, m in self._sources.items())
        except OSError:
            return
        if changed and self.server is not None:
            self.loop.create_task(self._reload_if_idle())

    async def _reload_if_idle(self):
        try:
            if await self.server.is_connected():
                return
        except Exception:
            pass
        log("bluetooth service: updated code: restarting")
        self.reload = True
        self._stop.set()


def _tidy(raw: str) -> str:
    """'2026-11-24T10:00:00-0500 python[123]: [  12.3s] message' -> '10:00:00 message'."""
    raw = raw.rstrip()
    if raw.startswith("-- "):                           # journalctl's own notes
        return ""
    if len(raw) > 19 and raw[4] == "-" and raw[10] == "T":
        clock, rest = raw[11:19], raw[19:]
        rest = rest.split(" ", 1)[1] if " " in rest else rest          # drop the UTC offset
        rest = rest.split(": ", 1)[1] if ": " in rest else rest        # drop 'python[123]'
        if rest.startswith("[") and "s] " in rest[:12]:                # the engine's own clock
            rest = rest.split("s] ", 1)[1]
        return f"{clock} {rest}"
    return raw


def main():
    svc = BleService()
    threading.Thread(target=svc.follow_journal, name="journal", daemon=True).start()
    svc.start()
    log("bluetooth service: running")
    try:
        while not svc._stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        svc.stop()
    finally:
        try:
            os.unlink(bc.BLE_SOCKET)
        except OSError:
            pass
    sys.exit(0)


if __name__ == "__main__":
    main()
