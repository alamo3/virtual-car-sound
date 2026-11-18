"""Tweaks on top of a voice -- what the sound lab (lab.py) edits and saves.

Two kinds of tweak, both stored in presets/<name>.json:

  muls   live multipliers on a voice's control targets (layer levels, frequencies,
         filter cutoffs...). Applied after every update(), so they act instantly. Targets
         named like o0, o1, o2... form one group ("o*") with a single multiplier.
  attrs  overrides of a voice class's numeric constants (F_IDLE, LEVEL, SPOOL_UP_SEC...).
         Some are read every tick (instant); the ones used while building the sound
         graph need the voice rebuilt -- `needs_rebuild` tells which.

main.py lists every saved preset as a voice of its own, so what you shape in the lab
is what plays in the car.
"""
from __future__ import annotations

import functools
import inspect
import json
import re
from pathlib import Path

import voices

PRESET_DIR = Path(__file__).parent / "presets"
FIXED = {"small", "master"}          # owned by the output stage and the effects ducking
SKIP_ATTRS = {"name", "kind"}
_GROUP = re.compile(r"^(.*?[a-z_])(\d+)$")


def group_of(key: str) -> str:
    m = _GROUP.match(key)
    return m.group(1) + "*" if m else key


def target_groups(voice) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key in voice.targets:
        if key not in FIXED:
            groups.setdefault(group_of(key), []).append(key)
    return groups


def voice_class(factory):
    """The class behind a VOICES entry (a class, or a partial that builds one)."""
    if isinstance(factory, type):
        return factory
    if isinstance(factory, functools.partial):
        if isinstance(factory.func, type):     # e.g. a resynth profile: the class itself
            return factory.func
        try:                                   # a mapped recording: which kind of voice?
            d = json.loads(Path(factory.args[0]).read_text())
            return voices.SampledRecording if d.get("samples") else voices.RecordingVoice
        except (OSError, ValueError, IndexError):
            return voices.RecordingVoice
    return None


def tunable_attrs(cls) -> dict[str, float]:
    """Numeric class constants a voice defines itself (not the shared Voice base)."""
    out = {}
    if cls is None:
        return out
    for klass in reversed(cls.__mro__):
        if klass in (object, voices.Voice):
            continue
        for k, v in vars(klass).items():
            if (k.isupper() and k not in SKIP_ATTRS and not k.startswith("_")
                    and isinstance(v, (int, float)) and not isinstance(v, bool)):
                out[k] = getattr(cls, k)
    return out


def needs_rebuild(cls, attr: str) -> bool:
    """True if the constant is used while building the graph (build / __init__)."""
    pat = re.compile(r"\bself\.%s\b" % re.escape(attr))
    for klass in cls.__mro__:
        if klass is object:
            continue
        for meth in ("build", "__init__"):
            fn = vars(klass).get(meth)
            if fn is None:
                continue
            try:
                if pat.search(inspect.getsource(fn)):
                    return True
            except (OSError, TypeError):
                return True
    return False


def attach(voice, muls: dict[str, float]):
    """Apply group multipliers after every update(). `voice.muls` can be edited live."""
    voice.muls = muls
    groups = target_groups(voice)
    base: dict[str, float] = {}
    applied: dict[str, float] = {}
    orig = voice.update

    def update(st):
        info = orig(st)
        for g, keys in groups.items():
            m = voice.muls.get(g, 1.0)
            for k in keys:
                tg = voice.targets[k]
                v = tg.value
                if k not in applied or v != applied[k]:
                    base[k] = v                    # update() set a fresh value this tick
                if m == 1.0 and k not in applied:
                    continue
                tg.set_value(base[k] * m)
                applied[k] = tg.value              # read back: Constants store float32
        return info

    voice.update = update
    return voice


def make_voice(factory, attrs=None, muls=None, name=None):
    """Build a voice with constant overrides and live multipliers."""
    attrs = dict(attrs or {})
    cls = voice_class(factory)
    if attrs and isinstance(factory, type):
        factory = type(factory.__name__ + "Tweaked", (factory,),
                       {**attrs, "trim_name": factory.name})
    v = factory()
    if name:
        v.name = name
    return attach(v, dict(muls or {}))


# --- files ----------------------------------------------------------------------------------

def preset_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9 _.-]+", "_", name).strip() or "preset"
    return PRESET_DIR / f"{safe}.json"


def save(name: str, voice: str, attrs: dict, muls: dict) -> Path:
    PRESET_DIR.mkdir(exist_ok=True)
    path = preset_path(name)
    clean_muls = {k: round(v, 4) for k, v in muls.items() if abs(v - 1.0) > 1e-4}
    path.write_text(json.dumps({"name": name, "voice": voice, "attrs": attrs,
                                "muls": clean_muls}, indent=2) + "\n")
    return path


def saved() -> list[dict]:
    if not PRESET_DIR.is_dir():
        return []
    out = []
    for p in sorted(PRESET_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            if "voice" in d:
                d.setdefault("name", p.stem)
                out.append(d)
        except (OSError, ValueError):
            pass
    return out


def register(registry: dict):
    """Add every saved preset to a VOICES-style registry as '<voice> · <preset>'."""
    for d in saved():
        base = registry.get(d["voice"])
        if base is None:
            continue
        label = f"{d['voice']} · {d['name']}"
        registry[label] = functools.partial(make_voice, base, d.get("attrs"), d.get("muls"),
                                            name=label)
