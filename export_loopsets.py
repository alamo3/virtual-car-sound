"""Export the loop-engine sound sets for apps that play them themselves (Open Blue).

    .venv/bin/python export_loopsets.py ~/open-blue-android/app/src/main/assets/loopsets

Each set is written exactly as loopengine.py prepares it before playback -- loops made
seamless (authored loop points or a tail crossfade), exterior loops EQ-matched to the
interior ones, all loops level-matched -- as mono 16-bit 48 kHz WAVs, plus set.json:

  {"name", "cylinders", "idle_rpm", "redline_rpm",
   "on":  [{"file", "ref_hz"}...],          on-load loops by rpm (firing frequency)
   "light": {"file", "ref_hz"} | null,       light-throttle loop
   "off": [{"file", "ref_hz"}...],          off-throttle loops by rpm
   "shift_up", "shift_down": file | null,   one-shots
   "pops": [file...]}                       lift-off / overrun pops
The firing frequency of a loop is rpm / 60 * cylinders / 2.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

import loopengine as le

SR = le.SR
# which sets to export, and the voice settings that go with them
EXPORT = {
    "bmw_m3_gtr_int": {"name": "BMW M3 GTR (interior)", "idle_rpm": 2200, "redline_rpm": 7000},
    "audi_r8_int": {"name": "Audi R8 (interior)", "idle_rpm": 2200, "redline_rpm": 7000,
                    "pitch": 0.75},
    "ioniq5n_int": {"name": "IONIQ 5 N (mod, interior)", "idle_rpm": 1000, "redline_rpm": 7500},
}


def write(path: Path, x: np.ndarray):
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).round().astype("<i2").tobytes())


def export(set_name: str, meta: dict, out: Path):
    s = le.SETS[set_name]
    d = out / set_name
    d.mkdir(parents=True, exist_ok=True)
    # on-load loops: same preparation as LoopEngine.build
    loops = [le._load(le._file(s, e[0]), seamless=True) for e in s["on"]]
    opts = [e[2] if len(e) > 2 else {} for e in s["on"]]
    native = [x for x, o in zip(loops, opts) if not o.get("match")]
    loops = [le._eq_match(x, native) if o.get("match") else x for x, o in zip(loops, opts)]
    rms = [float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) for x in loops]
    ref_rms = float(np.median(rms))
    if s.get("level_match"):
        loops = [x * (ref_rms / (r + 1e-9)) for x, r in zip(loops, rms)]
    # headroom: the loudest prepared loop peaks at 0.9
    peak = max(float(np.abs(x).max()) for x in loops)
    gain = 0.9 / peak if peak > 0.9 else 1.0
    pitch = meta.get("pitch", 1.0)
    on = []
    for e, x in zip(s["on"], loops):
        write(d / f"{e[0]}.wav", x * gain)
        on.append({"file": f"{e[0]}.wav", "ref_hz": e[1] / pitch})

    def single(entry):
        if not entry:
            return None
        fname, ref = entry
        write(d / f"{Path(fname).name}.wav", le._load(le._file(s, fname), seamless=True) * gain)
        return {"file": f"{Path(fname).name}.wav", "ref_hz": ref / pitch}

    offs = s.get("off_series") or ([s["off"]] if s.get("off") else [])
    off = [single(e) for e in offs]

    def oneshot(name):
        if not name:
            return None
        write(d / f"{Path(name).name}.wav", le._load(le._file(s, name)))
        return f"{Path(name).name}.wav"

    info = {"name": meta["name"], "cylinders": s["cylinders"], "idle_rpm": meta["idle_rpm"],
            "redline_rpm": meta["redline_rpm"], "on": on, "light": single(s.get("light")),
            "off": off, "shift_up": oneshot(s.get("shift_up")),
            "shift_down": oneshot(s.get("shift_down")),
            "pops": [oneshot(p) for p in s.get("pops", [])]}
    (d / "set.json").write_text(json.dumps(info, indent=1))
    size = sum(f.stat().st_size for f in d.iterdir()) / 1e6
    print(f"{set_name}: {len(on)} on, {len(off)} off, light {'yes' if info['light'] else 'no'}, "
          f"{len(info['pops'])} pops -> {d} ({size:.1f} MB)")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "loopsets").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name, meta in EXPORT.items():
        if le.available(name):
            export(name, meta, out)
        else:
            print(f"{name}: sound files missing, skipped")
    (out / "index.json").write_text(json.dumps(
        [{"id": n, "name": m["name"]} for n, m in EXPORT.items() if (out / n / "set.json").exists()],
        indent=1))


if __name__ == "__main__":
    main()
