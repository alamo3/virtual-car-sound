"""Extract the samples from an FMOD Studio .bank (its embedded FSB5 sound bank).

    .venv/bin/python fsb_extract.py path/to/car.bank            # -> sounds/extracted/<bank>/
    .venv/bin/python fsb_extract.py path/to/car.bank --list

Handles PCM (uncompressed) banks, which is what Assetto Corsa car mods usually ship.
Vorbis/FADPCM-encoded banks are reported, not decoded.
Game and mod audio is copyrighted: the extracted files are for personal use in this
project only -- don't commit, publish or share them.
"""
from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path

HERE = Path(__file__).parent
FREQ = {1: 8000, 2: 11000, 3: 11025, 4: 16000, 5: 22050, 6: 24000, 7: 32000, 8: 44100, 9: 48000}
CODECS = {1: "PCM8", 2: "PCM16", 3: "PCM24", 4: "PCM32", 5: "PCMFLOAT", 6: "GCADPCM",
          7: "IMAADPCM", 8: "VAG", 9: "HEVAG", 10: "XMA", 11: "MPEG", 12: "CELT", 13: "AT9",
          14: "XWMA", 15: "VORBIS", 16: "FADPCM", 17: "OPUS"}


def parse(data: bytes):
    base = data.find(b"FSB5")
    if base < 0:
        raise SystemExit("no FSB5 sound bank inside this file")
    ver, n, shdr_size, name_size, data_size, mode = struct.unpack_from("<6I", data, base + 4)
    hdr = 0x3C if ver == 1 else 0x40
    pos = base + hdr
    samples = []
    for _ in range(n):
        raw, = struct.unpack_from("<Q", data, pos)
        pos += 8
        more = raw & 1
        rate = FREQ.get((raw >> 1) & 0xF, 44100)
        ch = ((raw >> 5) & 1) + 1
        off = ((raw >> 6) & 0x0FFFFFFF) * 16
        frames = (raw >> 34) & 0x3FFFFFFF
        loop = None
        while more:
            c, = struct.unpack_from("<I", data, pos)
            pos += 4
            more, size, kind = c & 1, (c >> 1) & 0xFFFFFF, (c >> 25) & 0x7F
            if kind == 1:                                   # channel count override
                ch = data[pos]
            elif kind == 2:                                 # sample-rate override
                rate, = struct.unpack_from("<I", data, pos)
            elif kind == 3:                                 # loop points
                loop = struct.unpack_from("<2I", data, pos)
            pos += size
        samples.append({"rate": rate, "ch": ch, "off": off, "frames": frames, "loop": loop})
    names_at = base + hdr + shdr_size
    if name_size:
        for i, s in enumerate(samples):
            o, = struct.unpack_from("<I", data, names_at + 4 * i)
            end = data.index(b"\0", names_at + o)
            s["name"] = data[names_at + o:end].decode("utf-8", "replace")
    data_at = names_at + name_size
    for i, s in enumerate(samples):
        s.setdefault("name", f"sample_{i:03d}")
        nxt = samples[i + 1]["off"] if i + 1 < len(samples) else data_size
        s["bytes"] = data[data_at + s["off"]:data_at + nxt]
    return CODECS.get(mode, str(mode)), samples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bank")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bank = Path(a.bank)
    codec, samples = parse(bank.read_bytes())
    print(f"{bank.name}: {len(samples)} samples, {codec}")
    out = Path(a.out) if a.out else HERE / "sounds" / "extracted" / bank.stem
    if not a.list:
        if codec != "PCM16":
            raise SystemExit(f"{codec} banks aren't decoded by this script (PCM16 only)")
        out.mkdir(parents=True, exist_ok=True)
    for s in samples:
        sec = s["frames"] / s["rate"]
        loop = f"  loop {s['loop'][0]}-{s['loop'][1]}" if s["loop"] else ""
        print(f"  {s['name']:40s} {s['ch']}ch {s['rate']} Hz {sec:6.2f} s{loop}")
        if a.list:
            continue
        pcm = s["bytes"][:s["frames"] * s["ch"] * 2]
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in s["name"])
        with wave.open(str(out / f"{safe}.wav"), "wb") as w:
            w.setnchannels(s["ch"])
            w.setsampwidth(2)
            w.setframerate(s["rate"])
            w.writeframes(pcm)
    if not a.list:
        import json
        loops = {"".join(c if c.isalnum() or c in "._-" else "_" for c in x["name"]):
                 {"loop": list(x["loop"]) if x["loop"] else None, "rate": x["rate"],
                  "frames": x["frames"]} for x in samples}
        (out / "loops.json").write_text(json.dumps(loops, indent=1))
        print(f"wrote {len(samples)} files (+ loops.json with loop points) to {out}")


if __name__ == "__main__":
    main()
