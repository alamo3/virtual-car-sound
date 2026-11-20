"""Render the drive effects (fx.Effects) to sound files for Open Blue.

    .venv/bin/python export_effects.py ~/open-blue-android/app/src/main/assets/effects

The same effects the Pi plays, rendered with its own code (fx.py) instead of rewritten:
  launch.wav        the launch whoosh (impact + spool-up), one-shot
  shift_up.wav      the virtual-gear thump on an upshift, one-shot
  shift_down.wav    ... softer, on a downshift
  chime.wav         the power-on chime, one-shot
  reverse.wav       the reversing pad + ping, a seamless loop (10 s: the 0.9 Hz ping and
                    the 1.6 Hz pad pulse both complete whole cycles in it)
Mono 16-bit WAV at SR.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
BLOCK = 480


def write(path: Path, x: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


def trim_tail(x: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Drop the silent end of a one-shot (and fade its last 20 ms)."""
    loud = np.nonzero(np.abs(x) > 10 ** (floor_db / 20))[0]
    end = min(len(x), (loud[-1] + 1 + SR // 50) if len(loud) else len(x))
    y = x[:end].copy()
    n = min(len(y), SR // 50)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "effects").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from signalflow import AudioGraph, AudioGraphConfig
    cfg = AudioGraphConfig()
    cfg.backend_name = "null"
    cfg.sample_rate = SR
    cfg.output_buffer_size = BLOCK
    g = AudioGraph(cfg)
    g.stop()
    import fx
    e = fx.Effects()
    # connect every effect branch now and keep it connected, so the recordings don't depend
    # on when a branch is switched in (one-shots stay silent until fired, reverse until asked)
    for sec in e._sections.values():
        sec.hold(0.0, 1e9)
    dt = BLOCK / SR

    def run(sec, **kw):
        blocks = []
        for _ in range(int(round(sec / dt))):
            e.update(dt, **kw)
            buf = g.render_to_new_buffer(BLOCK)    # keep a reference: .data on a temporary
            blocks.append(np.array(buf.data[0], dtype=np.float64))   # reads freed memory
        return np.concatenate(blocks)

    run(0.5)                                    # settle the smoothers
    shots = {"launch": (e.launch, 2.6), "shift_up": (lambda: e.shift(up=True), 0.5),
             "shift_down": (lambda: e.shift(up=False), 0.5), "chime": (e.power_on, 4.8)}
    for name, (fire, sec) in shots.items():
        fire()
        x = trim_tail(run(sec))
        run(1.0)                                # let it ring out before the next one
        write(out / f"{name}.wav", x)
        print(f"{name}: {len(x) / SR:.2f} s, peak {np.abs(x).max():.2f}")

    run(3.0, reverse=True)                      # reverse: warm up, then one whole cycle
    loop = run(10.0, reverse=True)
    write(out / "reverse.wav", loop)
    print(f"reverse: {len(loop) / SR:.2f} s loop, peak {np.abs(loop).max():.2f}, "
          f"seam jump {abs(loop[0] - loop[-1]):.3f}")


if __name__ == "__main__":
    main()
