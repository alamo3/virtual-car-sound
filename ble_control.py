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
             testdrive start|stop  with --native-sound: a scripted drive (0-100-0 km/h)
             gearshifts 0|1        with --native-sound: the native engine's gearbox
             rev <0..1>|off        with --native-sound: rev in place (the app's Rev slider)
           (with --native-sound, "voice" picks one of the native sounds: Flat-6, Hypercar,
           Turbofan, Warp bubble)
  STATUS   read + notify, JSON, about once a second:
             {"v": voice, "vol": volume, "kmh": speed, "info": engine line,
              "car": data fresh?, "fx": .., "ss": .., "vg": .., "cq": .., "ih": ..,
              "msg": last problem, e.g. a voice that failed to load ("" when all is well),
              with --native-sound also "nat": {"gears": gearbox on, "g": gear, "rpm", "kmh",
              "lim": limiter, "td": test drive stage or "", "ur": underruns, "cpu": %,
              "live": car data, "cs": settled for a steady cruise} -- "nat" is {"gears"} alone
              until the player reports; the
              sounds are on LIST ("voices"))
  LIST     notify: the voice list, as above
  TELEMETRY  notify, as soon as each new reading arrives from the car (~25-30 a second;
           at least 25 a second even without one), 10 bytes little-endian -- the car's data,
           raw (not smoothed: the app smooths it itself, so it isn't delayed twice), for apps
           that make the sound themselves (Open Blue plays it through Android Auto):
             u8  version (1)
             u8  flags: bit0 data fresh, bit1 reversing
             u16 speed, km/h x 100
             u8  accelerator pedal, % x 2 (0..200)
             i16 motor rpm, signed (negative in reverse; 0 if unknown)
             i16 battery current, A x 10 (negative = regen / charging; 0 if unknown)
             u8  sequence number (wraps), to spot dropped packets

Every change is saved (settings.json), so the car starts the next drive the same way.
Runs on its own thread with its own asyncio loop; the audio loop reads the shared
Settings object each tick, so a change is picked up within ~20 ms (a voice switch builds
the new voice first, which can take a few seconds the first time).
"""
from __future__ import annotations

import asyncio
import json
import struct
import threading

SERVICE = "7a6b0001-3c1e-4b7a-9d55-2f1c0e5b8a01"
COMMAND = "7a6b0002-3c1e-4b7a-9d55-2f1c0e5b8a01"
STATUS = "7a6b0003-3c1e-4b7a-9d55-2f1c0e5b8a01"
LIST = "7a6b0004-3c1e-4b7a-9d55-2f1c0e5b8a01"
TELEMETRY = "7a6b0005-3c1e-4b7a-9d55-2f1c0e5b8a01"
TELEMETRY_HZ = 25
TELEMETRY_POLL_SEC = 0.004   # how often to look for a new reading to send
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

    RETRY_SEC = 10.0

    def _run(self):
        """Serve; if Bluetooth isn't there (adapter not up yet at boot, bluetoothd
        restarting, adapter unplugged), keep retrying every RETRY_SEC."""
        failures = 0
        while not self._stop.is_set():
            self.loop = asyncio.new_event_loop()
            try:
                self.loop.run_until_complete(self._serve())
                failures = 0                 # served until stop() or the link went away
            except Exception as e:           # no adapter, BlueZ down, no permission...
                if failures == 0 or failures % 30 == 0:     # don't flood the log
                    self.log(f"bluetooth control unavailable ({e}); retrying every "
                             f"{self.RETRY_SEC:.0f} s")
                failures += 1
            finally:
                if self.server is not None:      # unregister cleanly before retrying
                    try:
                        self.loop.run_until_complete(
                            asyncio.wait_for(self.server.stop(), timeout=5.0))
                    except Exception:
                        pass
                self.server = None
                self.loop.close()
            self._stop.wait(self.RETRY_SEC)

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
        await self.server.add_new_characteristic(
            SERVICE, TELEMETRY, C.read | C.notify, bytearray(self._telemetry()), P.readable)
        await self.server.start()
        self.log(f"bluetooth control: advertising as {NAME!r}")
        quiet, n = 0, 0
        loop_time = self.loop.time
        last_sent = last_status = loop_time()
        seen = None
        while not self._stop.is_set():
            # a new reading goes out straight away (checked every 4 ms); without one,
            # a packet at least TELEMETRY_HZ times a second so the app knows we're here
            await asyncio.sleep(TELEMETRY_POLL_SEC)
            now = loop_time()
            r = getattr(self.engine, "reader", None)
            stamp = getattr(r, "updated_at", None) if r is not None else None
            if stamp != seen or now - last_sent >= 1.0 / TELEMETRY_HZ:
                seen = stamp
                last_sent = now
                self._notify(TELEMETRY, self._telemetry())
            if now - last_status < 1.0:  # the rest runs once a second
                continue
            last_status = now
            self._notify(STATUS, self._status())
            n += 1
            if n % 10 == 0:             # health check: BlueZ restarted / adapter gone?
                # either call raises if bluetoothd went away -> _run starts over
                adv = await self.server.is_advertising()
                conn = await self.server.is_connected()
                quiet = 0 if (adv or conn) else quiet + 1
                if quiet >= 2:          # not advertising and nobody connected: re-register
                    raise RuntimeError("advertisement lost")
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
            if reply != "ok":
                self.engine.message = reply      # shown in the app
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
        if cmd == "testdrive":                   # the native player's scripted drive (start / stop)
            native = getattr(self.engine, "native", None)
            if native is None:
                return "no test drive: the Pi isn't playing the sound"
            native.command(f"testdrive {'stop' if arg.strip() == 'stop' else 'start'}")
            return "ok"
        if cmd in ("gearshifts", "rev"):
            native = getattr(self.engine, "native", None)
            if native is None:
                return f"no {cmd}: the Pi isn't playing the sound"
            if cmd == "rev":                     # momentary, not saved
                a = arg.strip()
                native.command("rev off" if a == "off" else f"rev {max(0.0, min(1.0, float(a))):.2f}")
                return None
            s.native_gears = arg.strip() in ("1", "true", "on")
        elif cmd == "voice":
            if arg not in self.voices:
                return f"unknown voice {arg!r}"
            if getattr(self.engine, "native", None) is not None:
                s.native_sound = arg
            else:
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

    _seq = 0

    def _telemetry(self) -> bytes:
        """The car's data as a 10-byte packet (layout in the module docstring)."""
        e = self.engine
        r = getattr(e, "reader", None)
        # the raw reading when there is one: the Pi's own smoothing is for its own sound
        raw_speed = getattr(r, "speed", None) if r else None
        raw_pedal = getattr(r, "pedal", None) if r else None
        live = bool(e.fresh) and raw_speed is not None and raw_pedal is not None
        speed = raw_speed if live else e.speed
        pedal = raw_pedal / 100.0 if live else getattr(e, "pedal_n", 0.0)
        rpm = getattr(r, "motor_rpm_signed", None) if r else None
        amps = getattr(r, "battery_current", None) if r else None
        events = getattr(e, "events", None)
        flags = (1 if e.fresh else 0) | (2 if events is not None and events.reverse else 0)
        self._seq = (self._seq + 1) & 0xFF
        clamp = lambda v, lo, hi: max(lo, min(hi, int(round(v))))
        return struct.pack("<BBHBhhB", 1, flags,
                           clamp(speed * 100, 0, 65535),
                           clamp(pedal * 200, 0, 200),
                           clamp(rpm or 0, -32768, 32767),
                           clamp((amps or 0) * 10, -32768, 32767),
                           self._seq)

    def _status(self) -> bytes:
        e, s = self.engine, self.engine.settings
        info = (e.info or "").strip()
        native = getattr(e, "native", None)
        st = {"v": s.native_sound if native is not None else s.voice, "vol": round(s.volume, 2), "kmh": round(e.speed, 1),
              "info": info[:60], "car": bool(e.fresh), "fx": s.effects, "ss": s.small_speaker,
              "vg": s.virtual_gears, "cq": s.cruise_quiet, "ih": round(s.idle_hum, 2),
              "msg": (getattr(e, "message", "") or "")[:120]}
        if native is not None:
            nat = {"gears": bool(s.native_gears)}
            p = native.status()
            if p:
                nat.update({k: p[k] for k in ("g", "rpm", "kmh", "lim", "td", "ur", "cpu", "live", "cs") if k in p})
            st["nat"] = nat
        return json.dumps(st, separators=(",", ":")).encode("utf-8")
