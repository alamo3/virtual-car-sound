"""Render what Open Blue's LoopVoice plays for a baked set, for listening on the desktop:
the same equal-power crossfades between neighbouring loops, pitch interpolation (refs),
cruise <-> accelerating blend by a glided load, and for spool sets the lagged spool and the
afterburner layer.

    .venv/bin/python preview_sets.py <assets dir> <out dir>
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
BLOCK = 480


def rd(p: Path) -> np.ndarray:
    """A file as (n, 2) stereo (mono doubled)."""
    with wave.open(str(p)) as w:
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, ch) / 32768.0
    return np.repeat(x, 2, axis=1) if ch == 1 else x[:, :2]


class Player:
    """One looping file, read at a variable rate (linear interpolation), like LoopVoice."""

    def __init__(self, x):
        self.x, self.pos = x, 0.0

    def block(self, rate, gain0, gain1):
        n = len(self.x)
        p = self.pos + rate * np.arange(BLOCK)
        i = np.floor(p).astype(int)
        f = p - i
        y = self.x[i % n] * (1 - f)[:, None] + self.x[(i + 1) % n] * f[:, None]
        self.pos = (self.pos + rate * BLOCK) % n
        return y * np.linspace(gain0, gain1, BLOCK)[:, None]


def render(set_dir: Path, script, seconds: float) -> np.ndarray:
    info = json.loads((set_dir / "set.json").read_text())
    grid = info["speeds"]
    series = {k: [(Player(rd(set_dir / e["file"])), e["ref_hz"]) for e in info.get(k, [])] for k in ("light", "on")}
    boost = Player(rd(set_dir / info["boost"]["file"])) if info.get("boost") else None
    spool_cfg = info.get("spool")
    gains = {}
    out = []
    load_g, spool, ab, speed_s = 0.0, None, 0.0, 0.0
    dt = BLOCK / SR
    for b in range(int(seconds / dt)):
        t = b * dt
        speed, pedal = script(t)
        speed_s += (speed - speed_s) * min(1, dt / 0.08)
        load_g += (pedal - load_g) * min(1, dt / 0.1)
        if spool_cfg:
            target = spool_cfg["idle_frac"] + (1 - spool_cfg["idle_frac"]) * load_g
            spool = target if spool is None else spool
            cap = (spool_cfg["up_per_sec"] if target > spool else spool_cfg["down_per_sec"]) * dt
            spool += float(np.clip((target - spool) * min(1, dt * 4), -cap, cap))
            v = spool
        else:
            v = speed_s
        if v <= grid[0]:
            i, u = 0, 0.0
        elif v >= grid[-1]:
            i, u = len(grid) - 2, 1.0
        else:
            i = max(k for k in range(len(grid) - 1) if grid[k] <= v)
            u = (v - grid[i]) / (grid[i + 1] - grid[i])
        wa, wb = np.cos(u * np.pi / 2), np.sin(u * np.pi / 2)
        if spool_cfg:
            beyond = 1.0
        elif v < grid[0]:
            beyond = 0.85 + 0.15 * min(1.0, max(0.0, v / grid[0]))
        elif v > grid[-1]:
            beyond = 1.0 + 0.1 * min(1.0, (v - grid[-1]) / 40)
        else:
            beyond = 1.0
        g_on = np.sin(load_g * np.pi / 2) if series["light"] else 1.0
        g_light = 1.0 if (spool_cfg and not series["on"]) else np.cos(load_g * np.pi / 2)
        y = np.zeros((BLOCK, 2))
        for name, g_series in (("on", g_on), ("light", g_light)):
            s = series[name]
            if not s:
                continue
            ref = np.exp(np.log(s[i][1]) * (1 - u) + np.log(s[i + 1][1]) * u)
            for k, (pl, r) in enumerate(s):
                g = g_series * (wa if k == i else wb if k == i + 1 else 0.0)
                key = (name, k)
                g0 = gains.get(key, 0.0)
                y += pl.block(ref * beyond / r, g0, g)
                gains[key] = g
        if boost:
            lit = load_g >= info["boost"].get("pedal", 2) and v > 0.9
            ab += ((1.0 if lit else 0.0) - ab) * min(1, dt / (0.4 if lit else 0.8))
            g0 = gains.get("boost", 0.0)
            y += boost.block(1.0, g0, ab)
            gains["boost"] = ab
        out.append(y)
    x = np.concatenate(out)
    return 0.9 * x / (np.abs(x).max() + 1e-9)


def write(p: Path, x):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2 if x.ndim == 2 else 1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


def main():
    assets, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ev = assets / "loopsets/future_ev"
    # 0 -> 140 km/h over 20 s, light pedal (cruise loops) / floored (accelerating loops)
    write(out / "future_ev_sweep_cruise.wav", render(ev, lambda t: (7 * t, 0.1), 20))
    write(out / "future_ev_sweep_accelerating.wav", render(ev, lambda t: (7 * t, 1.0), 20))
    # pedal on for 3 s, off for 3 s, while speeding up: the cruise <-> accelerating crossfade
    write(out / "future_ev_pedal_on_off.wav", render(ev, lambda t: (5 * t, 1.0 if int(t / 3) % 2 == 0 else 0.0), 24))
    jet = assets / "loopsets/fighter_jet"
    steps = lambda t: (40.0, 0.0 if t < 3 else 0.5 if t < 8 else 1.0 if t < 15 else 0.0)
    write(out / "fighter_jet_spool.wav", render(jet, steps, 22))
    print("written to", out)


if __name__ == "__main__":
    main()
