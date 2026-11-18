"""Bluetooth Low Energy control for the phone app (android/).

The Pi advertises a BLE service named "VirtualCarSound". No pairing is needed: the app
scans for the service, connects and talks over three characteristics:

  COMMAND  write, UTF-8 text, one command per write:
             voice <name>          switch voice
             volume <0..1>         master volume
             set <key> <value>     effects | small_speaker | virtual_gears | cruise_quiet
                                   (1/0), idle_hum (0..1)
             voices                send the voice list (on LIST, one name per notification,
                                   then "#end")
             status                send the status now
  STATUS   read + notify, JSON, about once a second:
             {"v": voice, "vol": volume, "kmh": speed, "info": engine line,
              "car": data fresh?, "fx": .., "ss": .., "vg": .., "cq": .., "ih": ..}
  LIST     notify: the voice list, as above

Every change is saved (settings.json), so the car starts the next drive the same way.
Runs on its own thread with its own asyncio loop; the audio loop reads the shared
Settings object each tick, so a change is picked up within ~20 ms (a voice switch builds
the new voice first, which can take a few seconds the first time).
"""
from __future__ import annotations

import asyncio
import json
import threading

SERVICE = "7a6b0001-3c1e-4b7a-9d55-2f1c0e5b8a01"
COMMAND = "7a6b0002-3c1e-4b7a-9d55-2f1c0e5b8a01"
STATUS = "7a6b0003-3c1e-4b7a-9d55-2f1c0e5b8a01"
LIST = "7a6b0004-3c1e-4b7a-9d55-2f1c0e5b8a01"
NAME = "VirtualCarSound"

BOOL_KEYS = {"effects", "small_speaker", "virtual_gears", "cruise_quiet"}
FLOAT_KEYS = {"idle_hum", "volume"}


class ControlServer:
    def __init__(self, engine, voice_names, save, log=print):
        self.engine = engine
        self.voices = list(voice_names)
        self.save = save                 # save(settings): persist after a change
        self.log = log
        self.loop: asyncio.AbstractEventLoop | None = None
        self.server = None
        self._stop = threading.Event()

    # --- thread / loop -------------------------------------------------------------
    def start(self):
        threading.Thread(target=self._run, name="ble-control", daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        self.loop = asyncio.new_event_loop()
        try:
            self.loop.run_until_complete(self._serve())
        except Exception as e:           # no adapter, BlueZ down, no permission...
            self.log(f"bluetooth control unavailable: {e}")

    async def _serve(self):
        from bless import (BlessServer, GATTAttributePermissions as P,
                           GATTCharacteristicProperties as C)
        self.server = BlessServer(name=NAME, loop=self.loop)
        self.server.read_request_func = self._on_read
        self.server.write_request_func = self._on_write
        await self.server.add_new_service(SERVICE)
        await self.server.add_new_characteristic(
            SERVICE, COMMAND, C.write | C.write_without_response, None, P.writeable)
        await self.server.add_new_characteristic(
            SERVICE, STATUS, C.read | C.notify, bytearray(self._status()), P.readable)
        await self.server.add_new_characteristic(
            SERVICE, LIST, C.read | C.notify, bytearray(b"#end"), P.readable)
        await self.server.start()
        self.log(f"bluetooth control: advertising as {NAME!r}")
        while not self._stop.is_set():
            self._notify(STATUS, self._status())
            await asyncio.sleep(1.0)
        await self.server.stop()

    # --- GATT callbacks (called on the loop's thread) -------------------------------
    def _on_read(self, characteristic, **_):
        if str(characteristic.uuid).lower() == STATUS:
            characteristic.value = bytearray(self._status())
        return characteristic.value

    def _on_write(self, characteristic, value, **_):
        if str(characteristic.uuid).lower() != COMMAND:
            return
        try:
            text = bytes(value).decode("utf-8").strip()
            reply = self.handle(text)
        except Exception as e:
            reply = f"error: {e}"
        if reply:
            self.log(f"phone: {text!r} -> {reply}")
        if text == "voices":
            self.loop.create_task(self._send_list())
        else:
            self._notify(STATUS, self._status())

    async def _send_list(self):
        for name in self.voices:
            self._notify(LIST, name.encode("utf-8")[:240])
            await asyncio.sleep(0.02)    # let each notification go out
        self._notify(LIST, b"#end")

    def _notify(self, uuid, data: bytes):
        if self.server is None:
            return
        ch = self.server.get_characteristic(uuid)
        if ch is None:
            return
        ch.value = bytearray(data)
        try:
            self.server.update_value(SERVICE, uuid)
        except Exception:
            pass                         # nobody subscribed yet

    # --- commands --------------------------------------------------------------------
    def handle(self, text: str) -> str | None:
        """Apply one command to the shared Settings; returns a short log line."""
        s = self.engine.settings
        parts = text.split(maxsplit=1)
        if not parts:
            return None
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        if cmd in ("voices", "status"):
            return None
        if cmd == "voice":
            if arg not in self.voices:
                return f"unknown voice {arg!r}"
            s.voice = arg
        elif cmd == "volume":
            s.volume = max(0.0, min(1.0, float(arg)))
        elif cmd == "set":
            key, _, val = arg.partition(" ")
            if key in BOOL_KEYS:
                setattr(s, key, val.strip() in ("1", "true", "on"))
            elif key in FLOAT_KEYS:
                setattr(s, key, max(0.0, min(1.0, float(val))))
            else:
                return f"unknown setting {key!r}"
        else:
            return f"unknown command {cmd!r}"
        self.save(s)
        return "ok"

    def _status(self) -> bytes:
        e, s = self.engine, self.engine.settings
        info = (e.info or "").strip()
        st = {"v": s.voice, "vol": round(s.volume, 2), "kmh": round(e.speed, 1),
              "info": info[:60], "car": bool(e.fresh), "fx": s.effects, "ss": s.small_speaker,
              "vg": s.virtual_gears, "cq": s.cruise_quiet, "ih": round(s.idle_hum, 2)}
        return json.dumps(st, separators=(",", ":")).encode("utf-8")
