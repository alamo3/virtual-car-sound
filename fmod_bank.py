"""Play Assetto Corsa FMOD car banks with the FMOD Studio runtime (desktop test).

    .venv/bin/python fmod_bank.py info   <car.bank>
    .venv/bin/python fmod_bank.py render <car.bank> out.wav [event ...]
    .venv/bin/python fmod_bank.py export <car.bank> <out dir> [--name NAME] [--idle RPM] [--redline RPM]
    .venv/bin/python fmod_bank.py export-jet "<plane>/fmod/Master Bank.bank" <out dir> <profile> [--name NAME]
    .venv/bin/python fmod_bank.py render-jet "<plane>/fmod/Master Bank.bank" out.wav <profile>

`info` loads the bank and lists its events with their parameters (names, ranges).
`export` packages a car for Open Blue: <out dir>/<car>/ with car.bank, master.bank (the
stand-in below) and sound.json (event IDs, parameter names, rev range, loudness). Copy the
folder to the phone and import it on the Sound screen. Banks stay off the public app: FMOD's
example media (the stand-in's source) and game mods may not be redistributed.
`render` plays the given events (default: engine_int) through a scripted drive -- launch,
pull through the gears, lift-off, coast -- with the Pi's drivetrain (drivetrain.py)
setting rpms / throttle / gear, and writes what FMOD mixes to a WAV, offline (FMOD's
non-realtime WAV writer: no sound card, faster than real time).

Needs the FMOD Engine for Linux (fmod.com; free account): FMOD_DIR points at the extracted
fmodstudioapiXXXXXlinux folder (default ~/Documents/fmod/fmodstudioapi20315linux).
GUIDs.txt next to the bank names the events (car banks carry no strings bank).

Master bank: AC car banks route to Assetto Corsa's master bus (bus:/, {08f6ba78-...} for
mods built with the usual AC SDK) and won't play without a bank defining it. The game's own
master bank isn't needed: master_bank() makes a stand-in by giving FMOD's example master bank
AC's master-bus ID. Events also start asynchronously (streams open on a loader thread in real
time), so the offline render waits for them and paces its updates a little.
"""
from __future__ import annotations

import ctypes as C
import os
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np

FMOD_DIR = Path(os.environ.get("FMOD_DIR", "~/Documents/fmod/fmodstudioapi20315linux")).expanduser()
FMOD_VERSION = 0x00020315
OUTPUT_WAVWRITER_NRT = 5
STUDIO_INIT_SYNCHRONOUS_UPDATE = 0x2
SR = 48000
BLOCK = 1024                  # FMOD's default mix block: one update = one block (NRT)
MASTER_DB = float(os.environ.get("MASTER_DB", "-16"))   # render level (see render())


class GUID(C.Structure):
    _fields_ = [("d1", C.c_uint32), ("d2", C.c_uint16), ("d3", C.c_uint16), ("d4", C.c_uint8 * 8)]

    @classmethod
    def parse(cls, s: str) -> "GUID":
        h = s.strip("{}").replace("-", "")
        b = bytes.fromhex(h)
        g = cls(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16))
        for i in range(8):
            g.d4[i] = b[8 + i]
        return g

    def __str__(self):
        d4 = bytes(self.d4).hex()
        return f"{{{self.d1:08x}-{self.d2:04x}-{self.d3:04x}-{d4[:4]}-{d4[4:]}}}"


class PARAM_ID(C.Structure):
    _fields_ = [("data1", C.c_uint32), ("data2", C.c_uint32)]


class PARAM_DESC(C.Structure):
    _fields_ = [("name", C.c_char_p), ("id", PARAM_ID), ("minimum", C.c_float),
                ("maximum", C.c_float), ("defaultvalue", C.c_float), ("type", C.c_int),
                ("flags", C.c_uint32), ("guid", GUID)]


def load_libs():
    core = C.CDLL(str(FMOD_DIR / "api/core/lib/x86_64/libfmod.so"), mode=C.RTLD_GLOBAL)
    studio = C.CDLL(str(FMOD_DIR / "api/studio/lib/x86_64/libfmodstudio.so"))
    return core, studio


class Fmod:
    def __init__(self, wav_out: str | None = None):
        self.core_lib, self.lib = load_libs()
        self.sys = C.c_void_p()
        self.ck(self.lib.FMOD_Studio_System_Create(C.byref(self.sys), FMOD_VERSION), "create")
        core = C.c_void_p()
        self.ck(self.lib.FMOD_Studio_System_GetCoreSystem(self.sys, C.byref(core)), "core")
        self.core = core
        if wav_out:
            self.ck(self.core_lib.FMOD_System_SetOutput(core, OUTPUT_WAVWRITER_NRT), "output")
        self.ck(self.core_lib.FMOD_System_SetSoftwareFormat(core, SR, 0, 0), "format")
        extra = C.c_char_p(wav_out.encode()) if wav_out else None
        self.ck(self.lib.FMOD_Studio_System_Initialize(
            self.sys, 256, STUDIO_INIT_SYNCHRONOUS_UPDATE, 0, extra), "initialize")

    @staticmethod
    def ck(r, what):
        if r != 0:
            raise RuntimeError(f"FMOD {what} failed: error {r}")

    def load_bank(self, path: str):
        bank = C.c_void_p()
        self.ck(self.lib.FMOD_Studio_System_LoadBankFile(
            self.sys, str(path).encode(), 0, C.byref(bank)), f"load {path}")
        # load all sample data now and wait for it: an offline render runs far faster than
        # real time and would otherwise finish before the background loader delivers it
        self.ck(self.lib.FMOD_Studio_Bank_LoadSampleData(bank), "load sample data")
        self.ck(self.lib.FMOD_Studio_System_FlushSampleLoading(self.sys), "flush sample loading")
        return bank

    def events(self, bank):
        n = C.c_int()
        self.ck(self.lib.FMOD_Studio_Bank_GetEventCount(bank, C.byref(n)), "event count")
        arr = (C.c_void_p * n.value)()
        got = C.c_int()
        self.ck(self.lib.FMOD_Studio_Bank_GetEventList(bank, arr, n.value, C.byref(got)), "event list")
        out = []
        for d in arr[:got.value]:
            g = GUID()
            self.ck(self.lib.FMOD_Studio_EventDescription_GetID(C.c_void_p(d), C.byref(g)), "id")
            out.append((C.c_void_p(d), str(g)))
        return out

    def params(self, desc):
        n = C.c_int()
        self.ck(self.lib.FMOD_Studio_EventDescription_GetParameterDescriptionCount(desc, C.byref(n)), "params")
        out = []
        for i in range(n.value):
            p = PARAM_DESC()
            self.ck(self.lib.FMOD_Studio_EventDescription_GetParameterDescriptionByIndex(
                desc, i, C.byref(p)), "param")
            out.append((p.name.decode(), p.minimum, p.maximum, p.defaultvalue))
        return out

    def instance(self, desc):
        inst = C.c_void_p()
        self.ck(self.lib.FMOD_Studio_EventDescription_CreateInstance(desc, C.byref(inst)), "instance")
        return inst

    def set(self, inst, name: str, value: float):
        return self.lib.FMOD_Studio_EventInstance_SetParameterByName(
            inst, name.encode(), C.c_float(value), 0)

    def start(self, inst):
        self.ck(self.lib.FMOD_Studio_EventInstance_Start(inst), "start")

    def update(self):
        self.ck(self.lib.FMOD_Studio_System_Update(self.sys), "update")

    def close(self):
        self.lib.FMOD_Studio_System_Release(self.sys)


AC_MASTER_BUS = "08f6ba78-f576-4f9f-a232-47657d5fec19"
# FMOD's example master bank (api/studio/examples/media): its master bus and the 11 group
# buses it defines, which master_bank() re-labels with the car's bus IDs
EXAMPLE_MASTER_BUS = "af9d027a-3a1f-49a8-a9ef-4cbe20673632"
EXAMPLE_GROUP_BUSES = [
    "d320eb98-3d4a-4cd9-a001-fdb4e071c58e", "37ebea86-d649-4a7a-b7b5-25701bfce8e9",
    "3e1d1653-8bd5-4dbc-865e-cba8e8ded970", "d9982c58-a056-4e6c-b8e3-883854b4bffb",
    "da23c66c-e512-4a93-98dc-02d6f58e4c5a", "69d0d47a-7f4b-4d6c-bfc3-bda91e9f7f1f",
    "dd124d8d-1ccd-4871-a160-5024e3e3f116", "7e33a2b3-db05-4b7a-ba1e-e862f2bfde36",
    "009ebac0-e5a9-4e69-8d37-6bdc9d8c6f45", "670390e5-527e-4f22-8654-b0326c9cc0c6",
    "524ae5a9-928c-4391-9635-34b41d4602af",
]
# AC group buses, most needed first (only the buses of events actually played must exist)
BUS_PRIORITY = ["engine_int", "gear_int", "limiter", "backfire_int", "transmission", "turbo",
                "engine_ext", "gear_ext", "backfire_ext", "wind", "wheel"]


def master_bank(bank_path: Path, out_dir: Path) -> Path | None:
    """A stand-in for Assetto Corsa's master bank: FMOD's example master bank with its buses
    re-labelled to the IDs the car bank routes to (from its GUIDs.txt) -- the master bus, and
    up to 11 group buses, the ones the engine sound uses first. None if it can't be made."""
    src = FMOD_DIR / "api/studio/examples/media/Master.bank"
    if not src.exists():
        return None
    data = src.read_bytes()
    car = bank_path.read_bytes()
    buses = {n: g.strip("{}") for g, n in guid_names(bank_path).items() if n.startswith("bus:/")}
    master = buses.get("bus:/", AC_MASTER_BUS)
    used = [n for n, g in buses.items() if n != "bus:/" and uuid.UUID(g).bytes_le in car]
    used.sort(key=lambda n: next((i for i, k in enumerate(BUS_PRIORITY) if k in n), 99))
    pairs = [(EXAMPLE_MASTER_BUS, master)] + list(zip(EXAMPLE_GROUP_BUSES, [buses[n] for n in used]))
    for old, new in pairs:
        data = data.replace(uuid.UUID(old).bytes_le, uuid.UUID(new).bytes_le)
    out = out_dir / f"master_for_{bank_path.stem}.bank"
    out.write_bytes(data)
    return out


def guid_names(bank_path: Path) -> dict[str, str]:
    txt = bank_path.parent / "GUIDs.txt"
    names = {}
    if txt.exists():
        for line in txt.read_text(errors="replace").splitlines():
            m = re.match(r"(\{[0-9a-fA-F-]+\})\s+(\S+)", line)
            if m:
                names[m.group(1).lower()] = m.group(2)
    return names


def info(bank_path: Path):
    f = Fmod()
    bank = f.load_bank(bank_path)
    names = guid_names(bank_path)
    evs = f.events(bank)
    print(f"{bank_path.name}: loaded, {len(evs)} events")
    for desc, gid in evs:
        ps = f.params(desc)
        print(f"  {names.get(gid.lower(), gid)}")
        for name, lo, hi, dflt in ps:
            print(f"      {name:20s} {lo:8.1f} .. {hi:8.1f}  (default {dflt:g})")
    f.close()


def drive_script(t: float) -> float:
    """Pedal over the test drive: idle, launch and pull through the gears, lift off and
    coast, cruise, a kickdown stab, then slow to a stop."""
    if t < 2.0:
        return 0.0
    if t < 13.0:
        return 1.0                       # flat out from rest
    if t < 18.0:
        return 0.0                       # lift off: overrun, pops, downshifts
    if t < 22.0:
        return 0.3                       # cruise
    if t < 25.0:
        return 1.0                       # kickdown
    return 0.0                           # coast down


def render(bank_path: Path, out: Path, event_names: list[str], seconds: float = 34.0,
           idle: float = 1100.0, redline: float = 8500.0, master_db: float | None = None) -> float:
    """Render the test drive; returns the rms of what FMOD mixed."""
    master_db = MASTER_DB if master_db is None else master_db
    import drivetrain
    import numpy as np
    f = Fmod(wav_out=str(out))
    mb = master_bank(bank_path, out.parent)
    if mb:
        m = C.c_void_p()
        f.ck(f.lib.FMOD_Studio_System_LoadBankFile(f.sys, str(mb).encode(), 0, C.byref(m)), "master bank")
    bank = f.load_bank(bank_path)
    if mb:
        # AC's mixer turns cars well down; on a bare master bus the events clip, so set the
        # master level here (MASTER_DB)
        bus = C.c_void_p()
        bid = GUID.parse(next((g for g, n in guid_names(bank_path).items() if n == "bus:/"),
                              "{" + AC_MASTER_BUS + "}"))
        if f.lib.FMOD_Studio_System_GetBusByID(f.sys, C.byref(bid), C.byref(bus)) == 0:
            f.lib.FMOD_Studio_Bus_SetVolume(bus, C.c_float(10 ** (master_db / 20)))
    names = guid_names(bank_path)
    by_name = {names.get(g.lower(), g).rsplit("/", 1)[-1]: d for d, g in f.events(bank)}
    loops = {}
    for n in event_names:
        if n not in by_name:
            raise SystemExit(f"no event {n!r}; have {sorted(by_name)}")
        inst = f.instance(by_name[n])
        f.start(inst)
        loops[n] = inst
    shots = {n: by_name[n] for n in ("gear_int", "limiter", "backfire_int") if n in by_name}
    # let the events start (async) before the drive; this silence is trimmed off below
    warm = 0
    for _ in range(200):
        f.update()
        warm += 1
        time.sleep(0.01)
        states = []
        for inst in loops.values():
            st = C.c_int()
            f.lib.FMOD_Studio_EventInstance_GetPlaybackState(inst, C.byref(st))
            states.append(st.value)
        if all(v == 0 for v in states):
            break
    else:
        raise SystemExit(f"events never started (states {states})")

    cfg = drivetrain.DrivetrainConfig(IDLE_RPM=idle, REDLINE_RPM=redline, LAUNCH_RPM=idle + 2400)
    dt_model = drivetrain.Drivetrain(cfg)
    dt = BLOCK / SR
    kmh, t, was_cut, was_on = 0.0, 0.0, False, False
    while t < seconds:
        pedal = drive_script(t)
        # simple car: pedal pushes, drag and rolling resistance pull back
        acc = pedal * 4.0 - 0.4 - 0.00012 * kmh * kmh - (1.5 if t >= 25.0 else 0.0)
        kmh = max(0.0, kmh + acc * 3.6 * dt)
        d = dt_model.step(kmh, pedal, dt)
        throttle = max(0.0, d.load)
        for inst in loops.values():
            f.set(inst, "rpms", d.rpm)
            f.set(inst, "throttle", throttle)
            f.set(inst, "drivetrain_speed", kmh)
        if d.event and "gear_int" in shots:
            g = f.instance(shots["gear_int"])
            f.set(g, "state", 0.0 if d.event == "up" else 1.0)
            f.start(g)
            f.lib.FMOD_Studio_EventInstance_Release(g)
        if d.cut and not was_cut and "limiter" in shots:
            g = f.instance(shots["limiter"])
            f.start(g)
            f.lib.FMOD_Studio_EventInstance_Release(g)
        if d.load < 0 and was_on and d.rpm_norm > 0.35 and "backfire_int" in shots:
            g = f.instance(shots["backfire_int"])
            f.set(g, "throttle", 0.0)
            f.start(g)
            f.lib.FMOD_Studio_EventInstance_Release(g)
        was_cut, was_on = d.cut, d.load > 0.2
        f.update()
        time.sleep(0.003)                # one-shots start asynchronously too
        t += dt
    f.close()
    import wave
    with wave.open(str(out)) as w:
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
        bits = w.getsampwidth() * 8
    x = (np.frombuffer(raw, "<f4") if bits == 32 else np.frombuffer(raw, "<i2") / 32768.0)
    x = x.reshape(-1, ch)[warm * BLOCK:]
    with wave.open(str(out), "wb") as w:            # rewrite without the warm-up
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())
    n = len(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    print(f"{out}: {n / sr:.1f} s, {ch} ch, {sr} Hz, {bits}-bit, peak {np.abs(x).max():.3f}, rms {rms:.4f}")
    return rms


# --- jets (X-Plane aircraft banks) ------------------------------------------------------
# An X-Plane aircraft's FMOD folder carries its own master bank; events take X-Plane datarefs
# as parameters. A car drives a jet like this: the pedal sets the engine-speed target and the
# engine spools toward it (slowly up, a bit faster down, like a turbofan); road speed is
# airspeed for the wind.
# Per-plane profiles: which looping events make the in-cockpit sound, each parameter's source,
# and how the engine spools. Sources: jet_rad / n1_pct (engine speed), throttle (pedal),
# speed_kts (road speed as airspeed), afterburner (lights near full pedal), const:<value>.
JET_PROFILES = {
    "s211": {
        # no side layer (nearly white noise: sounds like wind) and no cockpit wind (it
        # oscillates); the intake is kept low for its whine
        "loops": ["engine_out_rear", "engine_out_front"],
        "gains_db": {"engine_out_front": -10.0},
        "params": {
            "sim/flightmodel2/engines/engine_rotation_speed_rad_sec[0]": "jet_rad",
            "sim/cockpit2/engine/actuators/throttle_ratio[0]": "throttle",
            "sim/flightmodel/position/indicated_airspeed": "speed_kts",
        },
        "idle_frac": 0.55, "max_rad": 1520.0, "up_per_sec": 0.16, "down_per_sec": 0.22,
    },
    "f15a": {
        "loops": ["Engine", "EngineRocket", "Afterburner"],     # EngineJetLiner: whoosh
        # layer levels (dB): EngineRocket is the steadiest, most tonal layer; Engine and the
        # airliner layer swell up and down by themselves (heard as wind), and the airliner
        # layer is mostly sub-bass; the afterburner otherwise swamps the rest
        "gains_db": {"Engine": -12.0, "Afterburner": -20.0},   # afterburner: +4 dB at full power
        "params": {
            "sim/cockpit2/engine/indicators/N1_percent[*]": "n1_pct",
            "sim/flightmodel2/engines/afterburner_ratio[*]": "afterburner",
            "sim/cockpit2/gauges/indicators/airspeed_kts_pilot": "speed_kts",
            "xpevent_rel_heading": "const:180",
        },
        "idle_frac": 0.62, "max_rad": 1.0, "up_per_sec": 0.14, "down_per_sec": 0.2,
        "afterburner_pedal": 0.9,
        # launch (hard pedal stab from low speed): the plane's sonic boom. The event booms
        # ~2.5 s after Mach 1 (the shock reaching a ground observer), too late for a launch,
        # so it's rendered once to a sample (boom_sample) and played the moment you launch
        "boom": {"event": "SonicBoom", "param": "sim/flightmodel/misc/machno", "peak": 0.3},
    },
}
JET = JET_PROFILES["s211"]


class Launch:
    """The Pi's launch rule (fx.py): a fast, deep pedal stab from low speed, 4 s apart."""

    def __init__(self):
        self.prev, self.rate, self.cooldown = None, 0.0, 0.0

    def step(self, pedal, kmh, dt):
        if self.prev is not None and dt > 0:
            self.rate += ((pedal - self.prev) / dt - self.rate) * min(1.0, dt / 0.08)
        self.prev = pedal
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.rate > 2.0 and pedal > 0.65 and kmh < 50.0 and self.cooldown == 0.0:
            self.cooldown = 4.0
            return True
        return False


class Spool:
    """Turbofan spool: engine speed fraction chasing the pedal's target, rate-limited."""

    def __init__(self, idle_frac, up, down):
        self.idle, self.up, self.down = idle_frac, up, down
        self.n = idle_frac

    def step(self, pedal, dt):
        target = self.idle + (1.0 - self.idle) * pedal
        rate = self.up if target > self.n else self.down
        # quick near the target, rate-limited far from it: an exponential approach, capped
        step = (target - self.n) * min(1.0, dt * 2.5)
        cap = rate * dt
        self.n += max(-cap, min(cap, step))
        return self.n


def render_jet(bank_path: Path, out: Path, seconds: float = 34.0, master_db: float = -12.0,
               profile: dict | None = None) -> float:
    import numpy as np
    prof = profile or JET
    f = Fmod(wav_out=str(out))
    bank = f.load_bank(bank_path)
    strings = bank_path.parent / "Master Bank.strings.bank"
    if strings.exists():
        f.load_bank(strings)
    names = guid_names(bank_path)
    by_name = {names.get(g.lower(), g).rsplit("/", 1)[-1]: d for d, g in f.events(bank)}
    loops = {}
    for n in prof["loops"]:
        if n in by_name:
            inst = f.instance(by_name[n])
            loops[n] = (inst, [p[0] for p in f.params(by_name[n])])
            g = prof.get("gains_db", {}).get(n)
            if g is not None:
                f.lib.FMOD_Studio_EventInstance_SetVolume(inst, C.c_float(10 ** (g / 20)))
            f.start(inst)
    master = next((g for g, n in names.items() if n == "bus:/"), None)
    if master:
        bus = C.c_void_p()
        bid = GUID.parse(master)
        if f.lib.FMOD_Studio_System_GetBusByID(f.sys, C.byref(bid), C.byref(bus)) == 0:
            f.lib.FMOD_Studio_Bus_SetVolume(bus, C.c_float(10 ** (master_db / 20)))
    warm = 0
    for _ in range(200):
        f.update(); warm += 1; time.sleep(0.01)
        st = [C.c_int() for _ in loops]
        for (inst, _), s_ in zip(loops.values(), st):
            f.lib.FMOD_Studio_EventInstance_GetPlaybackState(inst, C.byref(s_))
        if all(x.value == 0 for x in st):
            break
    boom = prof.get("boom")
    launch = Launch()
    launches = []
    spool = Spool(prof["idle_frac"], prof["up_per_sec"], prof["down_per_sec"])
    dt = BLOCK / SR
    kmh, t, ab = 0.0, 0.0, 0.0
    while t < seconds:
        pedal = drive_script(t)
        acc = pedal * 4.0 - 0.4 - 0.00012 * kmh * kmh - (1.5 if t >= 25.0 else 0.0)
        kmh = max(0.0, kmh + acc * 3.6 * dt)
        n = spool.step(pedal, dt)
        if boom and launch.step(pedal, kmh, dt):
            launches.append(t)
            print(f"  launch at {t:.1f} s: sonic boom")
        lit = pedal >= prof.get("afterburner_pedal", 2.0) and n > 0.9
        ab += ((1.0 if lit else 0.0) - ab) * min(1.0, dt / (0.4 if lit else 0.8))
        values = {"jet_rad": n * prof["max_rad"], "n1_pct": n * 100.0, "throttle": pedal,
                  "speed_kts": kmh / 1.852, "afterburner": ab}
        for inst, params in loops.values():
            for p in params:
                src = prof["params"].get(p)
                if src is None:
                    continue
                f.set(inst, p, float(src[6:]) if src.startswith("const:") else values[src])
        f.update(); time.sleep(0.003); t += dt
    f.close()
    import wave
    with wave.open(str(out)) as w:
        ch, sr = w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, ch)[warm * BLOCK:] / 32768.0
    with wave.open(str(out), "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())
    if boom and launches:
        b = boom_sample(bank_path, boom)
        x = x.copy()
        for lt in launches:
            i = int(lt * sr)
            n = min(len(b), len(x) - i)
            x[i:i + n] += b[:n, None] * boom["peak"]
        with wave.open(str(out), "wb") as w:
            w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())
    rms = float(np.sqrt(np.mean(x ** 2)))
    print(f"{out}: {len(x) / sr:.1f} s, loops {list(loops)}, peak {np.abs(x).max():.3f}, rms {rms:.4f}")
    return rms


def boom_sample(bank_path: Path, boom: dict):
    """The plane's sonic boom as a mono sample (peak 1.0), starting at the crack: the event
    is run in (it can't boom for its first ~4 s), pushed through Mach 1, and recorded until
    it has rung out. A little saturation, kept above 150 Hz, gives the all-bass thud an edge
    small speakers can play."""
    import numpy as np
    import tempfile
    import wave
    tmp = Path(tempfile.mkdtemp()) / "boom.wav"
    f = Fmod(wav_out=str(tmp))
    bank = f.load_bank(bank_path)
    names = guid_names(bank_path)
    by_name = {names.get(g.lower(), g).rsplit("/", 1)[-1]: d for d, g in f.events(bank)}
    inst = f.instance(by_name[boom["event"]])
    f.set(inst, boom["param"], 0.0)
    f.set(inst, "xpevent_rel_heading", 0.0)
    f.start(inst)
    dt = BLOCK / SR
    for _ in range(int(4.5 / dt)):
        f.update(); time.sleep(0.002)
    for i in range(int(7.0 / dt)):
        f.set(inst, boom["param"], 0.95 + 0.35 * min(1.0, i * dt / 0.3))
        f.update(); time.sleep(0.002)
    f.close()
    with wave.open(str(tmp)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels()).mean(1) / 32768.0
    loud = np.nonzero(np.abs(x) > 0.02 * np.abs(x).max())[0]
    x = x[max(0, loud[0] - sr // 200):loud[-1] + sr // 10]
    x = x / (np.abs(x).max() + 1e-9)
    # edge: saturate, keep only what's above ~150 Hz of the distortion, add it back
    sat = np.tanh(3.0 * x)
    X = np.fft.rfft(sat); fr = np.fft.rfftfreq(len(sat), 1 / sr)
    X *= np.clip((fr - 100) / 100, 0, 1)
    x = x + 0.6 * np.fft.irfft(X, len(sat))
    fade = min(len(x), sr // 5)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x / (np.abs(x).max() + 1e-9)


# which events Open Blue plays, by the end of their name (AC's naming)
EXPORT_EVENTS = {"engine": "engine_int", "gear": "gear_int", "limiter": "limiter",
                 "backfire": "backfire_int"}
TARGET_RMS = 0.2
MASTER_DB_JET = 0.0       # export_jet measures loudness at 0 dB master


def export_jet(bank_path: Path, out_root: Path, profile: str, name: str | None):
    """Package an X-Plane aircraft's FMOD sound for Open Blue (sound.json format 2, kind
    "jet"): the aircraft's own master bank, the looping layers with their levels and where
    each parameter comes from, the spool model, and the launch boom as a sample."""
    import json
    import shutil
    import wave
    prof = JET_PROFILES[profile]
    names = guid_names(bank_path)
    f = Fmod()
    bank = f.load_bank(bank_path)
    descs = {names.get(g.lower(), g).rsplit("/", 1)[-1]: (d, g) for d, g in f.events(bank)}
    loops = []
    for n in prof["loops"]:
        d, g = descs[n]
        params = {p[0]: prof["params"][p[0]] for p in f.params(d) if p[0] in prof["params"]}
        loops.append({"name": n, "guid": g, "gain_db": prof.get("gains_db", {}).get(n, 0.0),
                      "params": params})
    f.close()
    sid = f"{profile}_{bank_path.parent.parent.name}".lower().replace(" ", "_")[:48]
    d = out_root / sid
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bank_path, d / "master.bank")
    # loudness: the test drive at 0 dB master -> gain to TARGET_RMS
    nob = dict(prof); nob.pop("boom", None)
    rms = render_jet(bank_path, d / "_level.wav", seconds=20.0, master_db=0.0, profile=nob)
    (d / "_level.wav").unlink()
    gain_db = float(np.clip(20 * np.log10(TARGET_RMS / max(rms, 1e-6)), -30, 30)) if rms else 0.0
    info = {"format": 2, "kind": "jet", "id": sid, "name": name or sid,
            "banks": ["master.bank"],
            "master_bus": next((g for g, n in names.items() if n == "bus:/"), None),
            "gain_db": round(gain_db, 1),
            "loops": loops,
            "jet": {k: prof[k] for k in ("idle_frac", "max_rad", "up_per_sec", "down_per_sec")}
                   | {"afterburner_pedal": prof.get("afterburner_pedal", 2.0)}}
    if prof.get("boom"):
        b = boom_sample(bank_path, prof["boom"])
        with wave.open(str(d / "boom.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((b * 0.98 * 32767).round().astype("<i2").tobytes())
        # the boom's level relative to the engine, as in the render (peak vs master -12 dB)
        info["boom"] = {"file": "boom.wav", "gain_db": round(20 * np.log10(prof["boom"]["peak"]) + 12.0 + gain_db - MASTER_DB_JET, 1)}
    (d / "sound.json").write_text(json.dumps(info, indent=1))
    print(f"{sid}: {[l['name'] for l in loops]}{' + boom' if 'boom' in info else ''} -> {d} (gain {gain_db:+.1f} dB)")


def export(bank_path: Path, out_root: Path, name: str | None, idle: float, redline: float):
    import json
    import shutil
    import tempfile
    names = guid_names(bank_path)
    f = Fmod()
    mb = master_bank(bank_path, Path(tempfile.mkdtemp()))
    if mb:
        m = C.c_void_p()
        f.ck(f.lib.FMOD_Studio_System_LoadBankFile(f.sys, str(mb).encode(), 0, C.byref(m)), "master")
    bank = f.load_bank(bank_path)
    events = {}
    for desc, gid in f.events(bank):
        short = names.get(gid.lower(), gid).rsplit("/", 1)[-1]
        for key, ev in EXPORT_EVENTS.items():
            if short == ev:
                events[key] = {"guid": gid, "params": [p[0] for p in f.params(desc)]}
    f.close()
    if "engine" not in events:
        raise SystemExit(f"{bank_path.name}: no {EXPORT_EVENTS['engine']} event")
    sid = bank_path.stem
    d = out_root / sid
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bank_path, d / "car.bank")
    shutil.copyfile(mb, d / "master.bank")
    # loudness: render the engine at 0 dB master and work out the gain to TARGET_RMS
    rms = render(bank_path, d / "_level.wav", ["engine_int"], seconds=20.0, idle=idle,
                 redline=redline, master_db=0.0)
    (d / "_level.wav").unlink()
    gain_db = float(np.clip(20 * np.log10(TARGET_RMS / max(rms, 1e-6)), -30, 30)) if rms else 0.0
    master_bus = next((g for g, n in names.items() if n == "bus:/"), "{" + AC_MASTER_BUS + "}")
    info = {"format": 1, "id": sid, "name": name or sid.replace("_", " "),
            "car_bank": "car.bank", "master_bank": "master.bank", "master_bus": master_bus,
            "idle_rpm": idle, "redline_rpm": redline, "gain_db": round(gain_db, 1),
            "events": events}
    (d / "sound.json").write_text(json.dumps(info, indent=1))
    print(f"{sid}: {sorted(events)} -> {d} (gain {gain_db:+.1f} dB)")


if __name__ == "__main__":
    cmd, bank = sys.argv[1], Path(sys.argv[2]).expanduser()
    if cmd == "info":
        info(bank)
    elif cmd == "render":
        render(bank, Path(sys.argv[3]).expanduser(), sys.argv[4:] or ["engine_int"])
    elif cmd == "render-jet":
        render_jet(bank, Path(sys.argv[3]).expanduser(),
                   profile=JET_PROFILES[sys.argv[4] if len(sys.argv) > 4 else "s211"])
    elif cmd == "export-jet":
        import argparse
        ap = argparse.ArgumentParser(prog="fmod_bank.py export-jet")
        ap.add_argument("out")
        ap.add_argument("profile", choices=sorted(JET_PROFILES))
        ap.add_argument("--name")
        a = ap.parse_args(sys.argv[3:])
        export_jet(bank, Path(a.out).expanduser(), a.profile, a.name)
    elif cmd == "export":
        import argparse
        ap = argparse.ArgumentParser(prog="fmod_bank.py export")
        ap.add_argument("out")
        ap.add_argument("--name")
        ap.add_argument("--idle", type=float, default=1100.0)
        ap.add_argument("--redline", type=float, default=8500.0)
        a = ap.parse_args(sys.argv[3:])
        export(bank, Path(a.out).expanduser(), a.name, a.idle, a.redline)
