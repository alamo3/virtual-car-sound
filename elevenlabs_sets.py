"""Build Open Blue sound sets and effects from ElevenLabs sound-effect generations.

    .venv/bin/python elevenlabs_sets.py ~/Downloads/elevenlabs_sfx ~/open-blue-android/app/src/main/assets [--pick group=take ...]

Input: the generations grouped in folders (several takes each), as prompted:
  futuristic_ev_engine/loop_layers/{low,medium,high,very_high}_speed      cruise loops
  futuristic_ev_engine/loop_layers/accelerating/{low,medium,high}_speed   full-pedal loops
  fighter_jet/engine_loop/{idle,mid,high}, fighter_jet/afterburner        jet loops
  cinematic gear shifts, launch, sonic_boom, reverse, startup             one-shots / loop
Output:
  loopsets/future_ev/   speed-keyed set (like the baked cinematic voices): cruise loops as
                        "light", accelerating loops as "on", a gentle pitch rise between speeds
  loopsets/fighter_jet/ spool-keyed set: idle / mid / high crossfaded by the jet's spool,
                        afterburner as a boost layer
  effects/              shift_cin_up/down, launch_ev, boom, reverse, chime
Takes are picked by measurement -- steady, clean (not noise-like), loopable, audible on small
speakers, and for loop series, sounding most alike from one speed to the next -- unless
--pick overrides (group = folder path, take = the "#n" in the file name).
Also writes previews of each pick to <assets>/../../../../elevenlabs_previews/ for listening.
"""
from __future__ import annotations

import json
import re
import sys
import wave
from itertools import product
from pathlib import Path

import numpy as np

SR = 48000
# loop levels (rms, dBFS): every loop of a series the same, so crossfades keep the level;
# accelerating loops a little above cruise, so the pedal adds weight without a jump
CRUISE_DB = -16.0
# pitch movement, so the sound glides with speed / spool rather than handing over between
# fixed loops: each loop is stretched toward its neighbour's pitch through the crossfade
EV_PITCH_STEP = 1.2             # per speed point (15, 50, 90, 130 km/h): x1.73 over the range
JET_PITCH = (0.85, 1.0, 1.18)   # idle, mid, high: the turbine winds up
ACCEL_DB = -14.0
JET_DB = -15.0


def load(p: Path) -> np.ndarray:
    """A file as stereo samples, shape (n, 2) at SR (mono files are doubled)."""
    with wave.open(str(p)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, "<i2") / 32768.0 if sw == 2 else np.frombuffer(raw, "<i4") / 2 ** 31
    x = x.reshape(-1, ch)
    x = np.repeat(x, 2, axis=1) if ch == 1 else x[:, :2]
    if sr != SR:
        n2 = int(round(len(x) * SR / sr))
        X = np.fft.rfft(x, axis=0)
        Y = np.zeros((n2 // 2 + 1, 2), complex)
        k = min(len(X), len(Y))
        Y[:k] = X[:k]
        x = np.fft.irfft(Y, n2, axis=0) * n2 / len(x)
    return x


def mono(x: np.ndarray) -> np.ndarray:
    return x.mean(1) if x.ndim == 2 else x


def write(p: Path, x: np.ndarray):
    """Write (n, 2) as a stereo 16-bit WAV (1-D as mono)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(x.shape[1] if x.ndim == 2 else 1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


def takes(folder: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(folder.glob("*.wav")):
        m = re.search(r"#(\d+)-(\d+)", p.name)
        key = f"#{m.group(1)}" if m else p.stem
        if key in out:                       # a second batch with the same take numbers
            key += "b"
        out[key] = p
    return out


def logspec(x: np.ndarray) -> np.ndarray:
    x = mono(x)
    n = 8192
    P = np.mean([np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n))) ** 2
                 for i in range(0, max(1, len(x) - n), n // 2)], 0)
    f = np.fft.rfftfreq(n, 1 / SR)
    grid = 60 * 2 ** (np.arange(0, np.log2(8000 / 60), 1 / 12))
    L = np.log(np.interp(grid, f, P) + 1e-12)
    return L - L.mean()


def quality(x: np.ndarray, loop: bool) -> float:
    """Lower is better: noise-likeness, unsteadiness, seam, and little audible content."""
    x = mono(x)
    n = 4096
    P = np.mean([np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n))) ** 2
                 for i in range(0, max(1, len(x) - n), n // 2)], 0)
    f = np.fft.rfftfreq(n, 1 / SR)
    b = (f > 80) & (f < 8000)
    flat = np.exp(np.mean(np.log(P[b] + 1e-20))) / np.mean(P[b])
    audible = P[(f >= 150) & (f <= 4000)].sum() / P.sum()
    whine = P[f > 4000].sum() / P.sum()
    score = 10 * flat + 2 * (1 - audible) + 20 * whine
    if loop:
        hop = SR // 50
        env = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))[::hop]
        core = env[len(env) // 10:-len(env) // 10]
        score += 0.5 * np.std(20 * np.log10(core + 1e-6))
    return float(score)


def pick_series(groups: list[dict[str, np.ndarray]], forced: list[str | None]) -> list[str]:
    """One take per group: good alone, and spectrally close to its neighbours (smooth
    crossfades). Exhaustive over the (small) combination space."""
    names = [list(g) if f is None else [f] for g, f in zip(groups, forced)]
    q = [{k: quality(g[k], True) for k in g} for g in groups]
    spec = [{k: logspec(g[k]) for k in g} for g in groups]
    best, best_s = None, 1e18
    for combo in product(*names):
        s = sum(q[i][k] for i, k in enumerate(combo))
        s += sum(np.mean((spec[i][a] - spec[i + 1][b]) ** 2) for i, (a, b) in enumerate(zip(combo, combo[1:])))
        if s < best_s:
            best, best_s = combo, s
    return list(best)


def seamless(x: np.ndarray, xfade: float = 0.3) -> np.ndarray:
    n = int(xfade * SR)
    # equal power (sin / cos): the end and the start are different sound, so an amplitude
    # crossfade (sin^2 / 1 - sin^2) would dip ~3 dB in the middle of every loop wrap
    a = np.linspace(0, np.pi / 2, n)
    if x.ndim == 2:
        a = a[:, None]
    y = x[:-n].copy()
    y[:n] = x[:n] * np.sin(a) + x[-n:] * np.cos(a)
    return y


def at_rms(x: np.ndarray, db: float) -> np.ndarray:
    """Scale a loop to an rms level (dBFS), so crossfades between loops don't change level;
    peaks are left to the app's limiter, but kept below full scale here."""
    y = x * 10 ** (db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    peak = np.abs(y).max()
    return y * (0.99 / peak) if peak > 0.99 else y


def trim_oneshot(x: np.ndarray, floor_db: float = -50.0) -> np.ndarray:
    a = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    thr = a.max() * 10 ** (floor_db / 20)
    idx = np.nonzero(a > thr)[0]
    x = x[max(0, idx[0] - SR // 400):idx[-1] + SR // 20].copy()
    fade = min(len(x) // 4, SR // 20)
    ramp = np.linspace(1, 0, fade)
    x[-fade:] *= ramp[:, None] if x.ndim == 2 else ramp
    return x / (np.abs(x).max() + 1e-9)


def hit_time(x: np.ndarray) -> float:
    """When a one-shot's main hit arrives: the first moment within 6 dB of its loudest."""
    x = mono(x)
    hop = SR // 100
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    return float(np.nonzero(env >= env.max() / 2)[0][0] * hop / SR)


def to_hit(x: np.ndarray, lead: float = 0.12) -> np.ndarray:
    """Cut a one-shot so its hit lands [lead] s in (a long swell would make it feel late),
    with a short fade-in where it was cut."""
    start = max(0, int((hit_time(x) - lead) * SR))
    y = x[start:].copy()
    n = min(len(y), int(0.03 * SR))
    ramp = np.linspace(0, 1, n)
    y[:n] *= ramp[:, None] if y.ndim == 2 else ramp
    return y


def edge(x: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """Add harmonics above ~150 Hz from a saturated copy: a bass-only hit gets an audible edge."""
    sat = np.tanh(3.0 * x / (np.abs(x).max() + 1e-9))
    X = np.fft.rfft(sat, axis=0)
    f = np.fft.rfftfreq(len(sat), 1 / SR)
    mask = np.clip((f - 100) / 100, 0, 1) / np.sqrt(1 + (f / 4000) ** 4)
    X *= mask[:, None] if X.ndim == 2 else mask
    y = x + amount * np.fft.irfft(X, len(sat), axis=0)
    return y / (np.abs(y).max() + 1e-9)


def main():
    src = Path(sys.argv[1]).expanduser()
    assets = Path(sys.argv[2]).expanduser()
    picks = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    previews = Path.home() / "Documents/fmod/renders/elevenlabs_previews"
    chosen = {}

    def group(rel):
        t = takes(src / rel)
        return {k: load(p) for k, p in t.items()}

    # --- Future EV: speed-keyed loop set -------------------------------------------------
    cruise_dirs = [f"futuristic_ev_engine/loop_layers/{s}_speed" for s in ("low", "medium", "high", "very_high")]
    accel_dirs = [f"futuristic_ev_engine/loop_layers/accelerating/{s}_speed" for s in ("low", "medium", "high")]
    speeds = [15, 50, 90, 130]
    cruise = [group(d) for d in cruise_dirs]
    accel = [group(d) for d in accel_dirs]
    cp = pick_series(cruise, [picks.get(d) for d in cruise_dirs])
    ap = pick_series(accel, [picks.get(d) for d in accel_dirs])
    chosen.update(zip(cruise_dirs, cp)); chosen.update(zip(accel_dirs, ap))
    d = assets / "loopsets/future_ev"
    info = {"name": "Future EV", "key": "speed", "speeds": speeds, "sample_rate": SR, "tone": False,
            "raw_level": True, "coast_is_light": True, "off": [], "pops": [], "light": [], "on": []}
    for i, (g, k) in enumerate(zip(cruise, cp)):
        write(d / f"cruise_{speeds[i]:03d}.wav", at_rms(seamless(g[k]), CRUISE_DB))
        info["light"].append({"file": f"cruise_{speeds[i]:03d}.wav", "ref_hz": EV_PITCH_STEP ** i})
    for i in range(len(speeds)):                 # no very-high accelerating take: reuse high
        g, k = accel[min(i, len(accel) - 1)], ap[min(i, len(accel) - 1)]
        write(d / f"accel_{speeds[i]:03d}.wav", at_rms(seamless(g[k]), ACCEL_DB))
        info["on"].append({"file": f"accel_{speeds[i]:03d}.wav", "ref_hz": EV_PITCH_STEP ** i})
    (d / "set.json").write_text(json.dumps(info, indent=1))

    # --- Fighter jet: spool-keyed loop set -----------------------------------------------
    jet_dirs = [f"fighter_jet/engine_loop/{s}" for s in ("idle", "mid", "high")]
    jet = [group(dj) for dj in jet_dirs]
    jp = pick_series(jet, [picks.get(dj) for dj in jet_dirs])
    ab = group("fighter_jet/afterburner")
    abk = picks.get("fighter_jet/afterburner") or min(ab, key=lambda k: quality(ab[k], True))
    chosen.update(zip(jet_dirs, jp)); chosen["fighter_jet/afterburner"] = abk
    d = assets / "loopsets/fighter_jet"
    info = {"name": "Fighter Jet", "key": "spool", "speeds": [0.0, 0.5, 1.0], "sample_rate": SR,
            "tone": False, "raw_level": True, "light": [], "on": [], "off": [], "pops": [],
            "spool": {"idle_frac": 0.0, "up_per_sec": 0.9, "down_per_sec": 1.1},
            "boost": {"file": "afterburner.wav", "ref_hz": 1.0, "pedal": 0.9}}
    for i, (g, k) in enumerate(zip(jet, jp)):
        name = ("idle", "mid", "high")[i]
        write(d / f"{name}.wav", at_rms(seamless(g[k]), JET_DB))
        info["light"].append({"file": f"{name}.wav", "ref_hz": JET_PITCH[i]})
    write(d / "afterburner.wav", at_rms(seamless(ab[abk]), JET_DB + 1.0))
    (d / "set.json").write_text(json.dumps(info, indent=1))

    # --- effects ------------------------------------------------------------------------
    fx = assets / "effects"
    sdir = src / "cinematic gear shifts"
    def by_take(pattern):
        out = {}
        for p in sorted(sdir.glob(pattern)):
            m = re.search(r"#(\d+)", p.name)
            out[f"#{m.group(1)}" if m else p.stem] = load(p)
        return out
    ups = by_take("*gear_shift*.wav")
    downs = by_take("*downshift*.wav")

    def best_oneshot(g, key, prefer_audible=True):
        if picks.get(key):
            return picks[key]
        def s(k):
            x = mono(g[k])
            P = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / SR)
            late = max(0.0, hit_time(x) - 0.5)          # a hit long after the trigger feels late
            return -P[(f >= 150) & (f <= 4000)].sum() / P.sum() + 20 * P[f > 5000].sum() / P.sum() + late
        return min(g, key=s)

    uk, dk = best_oneshot(ups, "shift_up"), best_oneshot(downs, "shift_down")
    chosen["shift_up"], chosen["shift_down"] = uk, dk
    write(fx / "shift_cin_up.wav", 0.85 * edge(trim_oneshot(to_hit(ups[uk])), 0.3))
    write(fx / "shift_cin_down.wav", 0.85 * edge(trim_oneshot(to_hit(downs[dk])), 0.3))
    launches = group("launch")
    lk = picks.get("launch") or min(launches, key=lambda k: quality(launches[k], False))
    chosen["launch"] = lk
    # the launch take builds for ~2 s, then bangs: split it so the app can play the build on
    # hard acceleration and add the bang only if the pedal is (nearly) floored at its end
    lx = launches[lk]
    lm = np.abs(mono(lx))
    lx = lx[np.nonzero(lm > lm.max() * 10 ** (-55 / 20))[0][0]:]
    bang_at = hit_time(lx)
    cut = max(0, int((bang_at - 0.03) * SR))
    build, bang = lx[:cut].copy(), lx[cut:].copy()
    fade = int(0.05 * SR)
    build[-fade:] *= np.linspace(1, 0, fade)[:, None]
    bang[:int(0.005 * SR)] *= np.linspace(0, 1, int(0.005 * SR))[:, None]
    write(fx / "launch_build.wav", 0.7 * edge(build / np.abs(build).max(), 0.8))
    write(fx / "launch_bang.wav", 0.9 * edge(trim_oneshot(bang), 0.6))
    print(f"  launch split: build {len(build) / SR:.2f} s, bang {len(bang) / SR:.2f} s")
    booms = group("sonic_boom")
    bk = best_oneshot(booms, "sonic_boom")
    chosen["sonic_boom"] = bk
    write(fx / "boom.wav", 0.9 * edge(trim_oneshot(booms[bk]), 0.4))
    rev = group("reverse")
    rk = picks.get("reverse") or "#1"
    chosen["reverse"] = rk
    write(fx / "reverse.wav", 0.7 * rev[rk] / np.abs(rev[rk]).max())        # already seamless
    st = group("startup")
    sk = picks.get("startup") or max((k for k in st if len(st[k]) > 6 * SR),
                                      key=lambda k: -quality(st[k], False))
    chosen["startup"] = sk
    write(fx / "chime.wav", 0.8 * trim_oneshot(st[sk], -60))

    # --- previews of the one-shots (preview_sets.py renders the loop sets as the app plays them)
    previews.mkdir(parents=True, exist_ok=True)
    for name in ("shift_cin_up", "shift_cin_down", "launch_build", "launch_bang", "boom", "chime"):
        x = load(fx / f"{name}.wav")
        write(previews / f"{name}.wav", x)
    (previews / "picks.json").write_text(json.dumps(chosen, indent=1))
    for k, v in chosen.items():
        print(f"  {k:55s} {v}")


if __name__ == "__main__":
    main()
