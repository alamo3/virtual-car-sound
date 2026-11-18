#!/usr/bin/env python3
"""Sound lab: shape any voice by ear while you drive it, and save the result as a preset.

    .venv/bin/python lab.py                 # starts on Dreadnought
    .venv/bin/python lab.py --voice "jet turbine"

Left: pick a voice and drive it -- hold Space (or Up) to floor it, let go to coast, or set
the pedal slider; "Hold speed" pins the speed so you can study one spot.
Right: every control of the voice as a slider.
  MIX        live multipliers (x1.00 = as designed) on the voice's layers, pitches and
             filters. Instant. Double-click a slider to reset it.
  STRUCTURE  the voice's design constants. Ones marked * rebuild the sound when you let
             go (a short gap is normal).
Mutate nudges every MIX slider at random by the amount set -- press it repeatedly and keep
what you like (Ctrl+Z undoes). Hold "Compare" to hear the voice without your MIX changes.
Save writes presets/<name>.json; main.py then lists it as "<voice> · <name>".
"""
from __future__ import annotations

import argparse
import math
import re
import random
import time
import tkinter as tk
from tkinter import messagebox, ttk
from types import SimpleNamespace

import main as engine
import presets
import voices

BG, PANEL, CARD, LINE = "#14161b", "#1c1f26", "#232731", "#2e3340"
TEXT, MUTED, ACCENT, CHANGED = "#e7e9ee", "#8b92a3", "#5b8cff", "#ffb454"
MUL_MAX = 3.0
HINT = ("Space/Up: floor it  ·  orange = changed  ·  double-click a slider to reset  ·  "
        "Ctrl+Z undo  ·  Ctrl+S save")
# pitches, rates and positions: mutate gently, never mute or double them
GENTLE = re.compile(r"(freq|_f$|_f\*|hz|rate|^sf|^rf|cut|bow|throb|pos|tone|band|root|"
                    r"cycle|note|^n\d|^f$|w_dec|dur)")


class LabSlider(tk.Canvas):
    """Flat slider with a mark at its default; double-click resets. `on_release` fires
    when the mouse is let go (for changes that rebuild the sound)."""

    def __init__(self, parent, lo, hi, value, default, step=0.01, command=None,
                 on_release=None, integer=False, on_press=None):
        super().__init__(parent, height=26, bg=CARD, highlightthickness=0, cursor="hand2")
        self.lo, self.hi, self.step, self.integer = lo, hi, step, integer
        self.value, self.default = value, default
        self.command, self.on_release, self.on_press = command, on_release, on_press
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda e: self.on_release and self.on_release())
        self.bind("<Double-Button-1>", self._reset)

    def _reset(self, _ev):
        if self.on_press:
            self.on_press()
        self.set(self.default, fire=True, release=True)

    def _x(self, v):
        return 10 + (v - self.lo) / (self.hi - self.lo) * (self.winfo_width() - 20)

    def _press(self, ev):
        if self.on_press:
            self.on_press()
        self._drag(ev)

    def _drag(self, ev):
        v = self.lo + (ev.x - 10) / max(1, self.winfo_width() - 20) * (self.hi - self.lo)
        self.set(v, fire=True)

    def set(self, v, fire=False, release=False):
        v = min(self.hi, max(self.lo, v))
        v = round(v) if self.integer else round(v / self.step) * self.step
        changed = v != self.value
        self.value = v
        self.draw()
        if fire and changed and self.command:
            self.command(v)
        if release and self.on_release:
            self.on_release()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        y = h / 2
        x, xd = self._x(self.value), self._x(self.default)
        off = abs(self.value - self.default) > 1e-9
        self.create_line(10, y, w - 10, y, fill=LINE, width=4, capstyle="round")
        self.create_line(xd, y, x, y, fill=CHANGED if off else ACCENT, width=4)
        self.create_line(xd, y - 7, xd, y + 7, fill=MUTED)
        self.create_oval(x - 7, y - 7, x + 7, y + 7, fill="white",
                         outline=CHANGED if off else ACCENT, width=2)


class Lab:
    def __init__(self, root, voice, null_audio=False):
        self.root = root
        engine.MAX_MOTOR_RPM = engine.CAR_RPM_MAX
        if null_audio:
            cfg = engine.AudioGraphConfig()
            cfg.backend_name = "null"
            self.graph = engine.AudioGraph(cfg)
        else:
            self.graph = engine.make_graph(SimpleNamespace(output_device=None, buffer_size=None))
        self.powertrain = engine.Powertrain(0.0)
        self.swell = engine.TipInSwell()
        self.pedal = tk.DoubleVar(value=0.0)
        self.hold = tk.BooleanVar(value=False)
        self.small = tk.BooleanVar(value=voices.SMALL_SPEAKER)
        self.floored = False
        self._release_job = None
        self.voice = None
        self.attrs: dict[str, float] = {}
        self.muls: dict[str, float] = {}
        self.history: list[tuple[dict, dict]] = []
        self.last = time.monotonic()
        self._style()
        self._build()
        self._bind()
        self.load_voice(voice)
        self.tick()

    # -- look ----------------------------------------------------------------------------
    def _style(self):
        r = self.root
        r.title("Sound lab")
        r.configure(bg=BG)
        r.geometry("1180x800")
        r.minsize(980, 640)
        s = ttk.Style(r)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, fieldbackground=CARD,
                    font=("TkDefaultFont", 10), bordercolor=LINE, lightcolor=LINE,
                    darkcolor=LINE, troughcolor=CARD, focuscolor=ACCENT, insertcolor=TEXT)
        for name, bg in (("Panel", PANEL), ("Card", CARD)):
            s.configure(f"{name}.TFrame", background=bg)
            s.configure(f"{name}.TLabel", background=bg, foreground=TEXT)
            s.configure(f"{name}.TCheckbutton", background=bg, foreground=TEXT)
            s.map(f"{name}.TCheckbutton", background=[("active", bg)])
        s.configure("Muted.Card.TLabel", background=CARD, foreground=MUTED,
                    font=("TkDefaultFont", 9))
        s.configure("Changed.Card.TLabel", background=CARD, foreground=CHANGED,
                    font=("TkDefaultFont", 9))
        s.configure("CardTitle.TLabel", background=CARD, foreground=MUTED,
                    font=("TkDefaultFont", 9, "bold"))
        s.configure("Title.TLabel", background=PANEL, font=("TkDefaultFont", 14, "bold"))
        s.configure("Big.Card.TLabel", background=CARD, font=("TkDefaultFont", 26, "bold"))
        s.configure("Status.TLabel", background=PANEL, foreground=MUTED, padding=(12, 6))
        s.configure("TButton", background=CARD, foreground=TEXT, padding=(10, 5),
                    borderwidth=1, relief="flat")
        s.map("TButton", background=[("active", LINE)])
        s.configure("Accent.TButton", background=ACCENT, foreground="white")
        s.map("Accent.TButton", background=[("active", "#7aa2ff")])
        s.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=TEXT,
                    arrowcolor=TEXT, padding=(6, 4))
        s.map("TCombobox", fieldbackground=[("readonly", CARD)], foreground=[("readonly", TEXT)],
              background=[("readonly", CARD), ("active", LINE)],
              selectbackground=[("readonly", CARD)], selectforeground=[("readonly", TEXT)])
        r.option_add("*TCombobox*Listbox.background", CARD)
        r.option_add("*TCombobox*Listbox.foreground", TEXT)

    def _card(self, parent, title):
        f = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12))
        f.pack(fill="x", pady=(0, 10))
        ttk.Label(f, text=title.upper(), style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        return f

    def _build(self):
        head = ttk.Frame(self.root, style="Panel.TFrame", padding=(14, 10))
        head.pack(fill="x")
        ttk.Label(head, text="Sound lab", style="Title.TLabel").pack(side="left")
        self.voice_var = tk.StringVar()
        names = [n for n in engine.VOICES if " · " not in n]
        cb = ttk.Combobox(head, textvariable=self.voice_var, values=names, state="readonly",
                          width=30)
        cb.pack(side="left", padx=16)
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_voice(self.voice_var.get()))
        self.preset_var = tk.StringVar()
        self.preset_cb = ttk.Combobox(head, textvariable=self.preset_var, state="readonly",
                                      width=24)
        self.preset_cb.pack(side="left")
        self.preset_cb.bind("<<ComboboxSelected>>", lambda e: self.load_preset())
        ttk.Button(head, text="Save preset…", style="Accent.TButton",
                   command=self.save).pack(side="right")

        self.status = ttk.Label(self.root, text=HINT, style="Status.TLabel", anchor="w")
        self.status.pack(side="bottom", fill="x")

        body = ttk.Frame(self.root, padding=12)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, width=330)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        drive = self._card(left, "Drive")
        self.speed_lbl = ttk.Label(drive, text="0 km/h", style="Big.Card.TLabel")
        self.speed_lbl.pack(anchor="w")
        self.info_lbl = ttk.Label(drive, text="", style="Muted.Card.TLabel")
        self.info_lbl.pack(anchor="w", pady=(0, 8))
        ttk.Label(drive, text="Pedal  (hold Space / Up to floor it)",
                  style="Muted.Card.TLabel").pack(anchor="w")
        self.pedal_slider = LabSlider(drive, 0.0, 1.0, 0.0, 0.0, step=0.01,
                                      command=lambda v: self.pedal.set(v))
        self.pedal_slider.pack(fill="x")
        ttk.Checkbutton(drive, text="Hold speed at", variable=self.hold,
                        style="Card.TCheckbutton").pack(anchor="w", pady=(8, 0))
        self.hold_slider = LabSlider(drive, 0.0, 180.0, 50.0, 50.0, step=1.0,
                                     command=lambda v: self.hold.set(True))
        self.hold_slider.pack(fill="x")
        ttk.Checkbutton(drive, text="Small speaker mode", variable=self.small,
                        style="Card.TCheckbutton",
                        command=lambda: self.voice.set_small_speaker(self.small.get())
                        ).pack(anchor="w", pady=(8, 0))

        explore = self._card(left, "Explore")
        ttk.Label(explore, text="Mutate amount", style="Muted.Card.TLabel").pack(anchor="w")
        self.amount = LabSlider(explore, 0.05, 1.0, 0.3, 0.3, step=0.05)
        self.amount.pack(fill="x")
        self.mutate_structure = tk.BooleanVar(value=False)
        ttk.Checkbutton(explore, text="Mutate structure too",
                        variable=self.mutate_structure,
                        style="Card.TCheckbutton").pack(anchor="w", pady=(4, 6))
        row = ttk.Frame(explore, style="Card.TFrame")
        row.pack(fill="x")
        for i, (text, cmd, style) in enumerate((("Mutate", self.mutate, "Accent.TButton"),
                                                 ("Undo", self.undo, "TButton"),
                                                 ("Reset", self.reset, "TButton"))):
            ttk.Button(row, text=text, command=cmd, style=style).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
            row.columnconfigure(i, weight=1)
        cmp_btn = ttk.Button(explore, text="Hold to compare with original")
        cmp_btn.pack(fill="x", pady=(8, 0))
        cmp_btn.bind("<ButtonPress-1>", lambda e: self.compare(True))
        cmp_btn.bind("<ButtonRelease-1>", lambda e: self.compare(False))


        # scrollable parameter area
        right = ttk.Frame(body, style="Card.TFrame")
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, bg=CARD, highlightthickness=0)
        sb = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        self.params = ttk.Frame(self.canvas, style="Card.TFrame", padding=(14, 12))
        self.params.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.params, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win,
                                                                            width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for seq, d in (("<Button-4>", -3), ("<Button-5>", 3)):
            self.canvas.bind_all(seq, lambda e, d=d: self.canvas.yview_scroll(d, "units"))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(
            -1 if e.delta > 0 else 1, "units"))

    def _bind(self):
        for key in ("<KeyPress-space>", "<KeyPress-Up>"):
            self.root.bind(key, self._floor_on)
        for key in ("<KeyRelease-space>", "<KeyRelease-Up>"):
            self.root.bind(key, self._floor_off)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # held keys auto-repeat as press/release pairs on X11: only act on a real release
    def _floor_on(self, _ev):
        if self._release_job:
            self.root.after_cancel(self._release_job)
            self._release_job = None
        self.floored = True

    def _floor_off(self, _ev):
        self._release_job = self.root.after(40, self._really_off)

    def _really_off(self):
        self._release_job = None
        self.floored = False

    # -- voice and parameters ----------------------------------------------------------
    def load_voice(self, name, attrs=None, muls=None):
        self.voice_name = engine.find_voice(name)
        self.voice_var.set(self.voice_name)
        self.factory = engine.VOICES[self.voice_name]
        self.cls = presets.voice_class(self.factory)
        self.attrs = dict(attrs or {})
        self.muls = dict(muls or {})
        self.history.clear()
        self._rebuild()
        self._populate()
        names = [d["name"] for d in presets.saved() if d["voice"] == self.voice_name]
        self.preset_cb.configure(values=names)
        self.preset_var.set("")

    def _rebuild(self):
        t = time.monotonic()
        new = presets.make_voice(self.factory, self.attrs if self.cls else None, self.muls)
        new.set_small_speaker(self.small.get())
        new.output.play()
        if self.voice is not None:
            self.voice.output.stop()
        self.voice = new
        self._flash(f"built {self.voice_name} in {time.monotonic() - t:.2f} s")

    def _populate(self):
        for w in self.params.winfo_children():
            w.destroy()
        self.mix_sliders, self.attr_sliders = {}, {}
        ttk.Label(self.params, text="MIX  (×1.00 = as designed)",
                  style="CardTitle.TLabel").pack(anchor="w", pady=(0, 6))
        for g, keys in presets.target_groups(self.voice).items():
            label = g if len(keys) == 1 else f"{g}  ({len(keys)})"
            self.mix_sliders[g] = self._row(
                label, 0.0, MUL_MAX, self.muls.get(g, 1.0), 1.0,
                command=lambda v, g=g: self._set_mul(g, v), fmt="×{:.2f}")
        attrs = presets.tunable_attrs(self.cls)
        if attrs:
            ttk.Label(self.params, text="STRUCTURE  (* = rebuilds on release)",
                      style="CardTitle.TLabel").pack(anchor="w", pady=(14, 6))
        for k, default in attrs.items():
            rebuild = presets.needs_rebuild(self.cls, k)
            integer = isinstance(default, int)
            lo, hi = (0.0, default * 2.0) if default > 0 else (default * 2.0, 0.0) \
                if default < 0 else (-1.0, 1.0)
            if integer:
                lo, hi = max(1, int(lo)), max(int(hi), 2)
            step = 1 if integer else _nice_step(hi - lo)
            self.attr_sliders[k] = self._row(
                k + (" *" if rebuild else ""), lo, hi, self.attrs.get(k, default), default,
                command=lambda v, k=k, r=rebuild: self._set_attr(k, v, r),
                on_release=(self._rebuild_now if rebuild else None), integer=integer,
                fmt="{:.0f}" if integer else "{:.4g}", step=step)

    def _row(self, label, lo, hi, value, default, command, on_release=None, integer=False,
             fmt="{:.2f}", step=0.01):
        row = ttk.Frame(self.params, style="Card.TFrame")
        row.pack(fill="x", pady=1)
        name = ttk.Label(row, text=label, width=22, style="Card.TLabel", anchor="w")
        name.pack(side="left")
        val = ttk.Label(row, width=9, style="Muted.Card.TLabel", anchor="e")
        val.pack(side="right")

        def show(v):
            changed = abs(v - default) > 1e-9
            val.configure(text=fmt.format(v),
                          style="Changed.Card.TLabel" if changed else "Muted.Card.TLabel")

        def cmd(v):
            show(v)
            command(v)

        s = LabSlider(row, lo, hi, value, default, step=step, command=cmd,
                      on_release=on_release, integer=integer, on_press=self._snapshot)
        s.pack(side="left", fill="x", expand=True, padx=8)
        s.show = show
        show(value)
        return s

    def _snapshot(self):
        self.history.append((dict(self.attrs), dict(self.muls)))
        del self.history[:-50]

    def _set_mul(self, g, v):
        self.muls[g] = v
        self.voice.muls = self.muls

    def _set_attr(self, k, v, rebuild):
        self.attrs[k] = v
        if not rebuild:
            setattr(self.voice, k, v)          # read every tick: takes effect now

    def _rebuild_now(self):
        self._rebuild()

    # -- exploring -----------------------------------------------------------------------
    def mutate(self):
        self._snapshot()
        amt = self.amount.value
        for g, s in self.mix_sliders.items():
            if GENTLE.search(g):                      # pitch/rate/position: small nudges
                v = self.muls.get(g, 1.0) * math.exp(random.gauss(0.0, amt * 0.25))
                v = min(1.6, max(0.6, v))
            else:
                v = self.muls.get(g, 1.0) * math.exp(random.gauss(0.0, amt))
                if random.random() < 0.08 * amt:      # occasionally mute or slam a layer
                    v = random.choice((0.0, 2.0))
            v = min(MUL_MAX, max(0.0, v))
            self.muls[g] = v
            s.set(v)
            s.show(s.value)
        rebuild = False
        if self.mutate_structure.get():
            for k, s in self.attr_sliders.items():
                if random.random() < 0.5:
                    continue
                v = s.value * math.exp(random.gauss(0.0, amt * 0.6)) if s.value else \
                    random.uniform(s.lo, s.hi)
                s.set(v)
                s.show(s.value)
                self.attrs[k] = s.value
                if presets.needs_rebuild(self.cls, k):
                    rebuild = True
                else:
                    setattr(self.voice, k, s.value)
        self.voice.muls = self.muls
        if rebuild:
            self._rebuild()
        self._flash("mutated -- Ctrl+Z to undo")

    def undo(self):
        if not self.history:
            self._flash("nothing to undo")
            return
        attrs, muls = self.history.pop()
        rebuild = any(presets.needs_rebuild(self.cls, k) for k in set(attrs) | set(self.attrs)
                      if attrs.get(k) != self.attrs.get(k))
        self.attrs, self.muls = attrs, muls
        if rebuild:
            self._rebuild()
        else:
            for k, v in attrs.items():
                setattr(self.voice, k, v)
        self.voice.muls = self.muls
        self._sync_sliders()
        self._flash("undone")

    def reset(self):
        self._snapshot()
        rebuild = any(presets.needs_rebuild(self.cls, k) for k in self.attrs)
        self.attrs, self.muls = {}, {}
        if rebuild:
            self._rebuild()
        else:
            for k in presets.tunable_attrs(self.cls):
                if k in vars(self.voice):
                    delattr(self.voice, k)
        self.voice.muls = self.muls
        self._sync_sliders()

    def _sync_sliders(self):
        for g, s in self.mix_sliders.items():
            s.set(self.muls.get(g, 1.0))
            s.show(s.value)
        for k, s in self.attr_sliders.items():
            s.set(self.attrs.get(k, s.default))
            s.show(s.value)

    def compare(self, on):
        self.voice.muls = {} if on else self.muls
        self._flash("original" if on else "your version")

    # -- presets -------------------------------------------------------------------------
    def save(self):
        win = tk.Toplevel(self.root, bg=PANEL)
        win.title("Save preset")
        win.transient(self.root)
        ttk.Label(win, text=f"Preset name for {self.voice_name}:",
                  style="Panel.TLabel").pack(padx=14, pady=(14, 6), anchor="w")
        var = tk.StringVar(value=self.preset_var.get() or "my version")
        e = ttk.Entry(win, textvariable=var, width=34)
        e.pack(padx=14)
        e.focus_set()
        e.select_range(0, "end")

        def ok(_ev=None):
            name = var.get().strip()
            if not name:
                return
            path = presets.save(name, self.voice_name,
                                {k: v for k, v in self.attrs.items()}, self.muls)
            win.destroy()
            names = [d["name"] for d in presets.saved() if d["voice"] == self.voice_name]
            self.preset_cb.configure(values=names)
            self.preset_var.set(name)
            self._flash(f"saved {path.name} -- main.py lists it as "
                        f"\"{self.voice_name} · {name}\" (restart main.py)")

        ttk.Button(win, text="Save", style="Accent.TButton", command=ok).pack(pady=12)
        win.bind("<Return>", ok)

    def load_preset(self):
        name = self.preset_var.get()
        for d in presets.saved():
            if d["voice"] == self.voice_name and d["name"] == name:
                self.load_voice(self.voice_name, d.get("attrs"), d.get("muls"))
                self.preset_var.set(name)
                self._flash(f"loaded preset {name}")
                return

    # -- running -------------------------------------------------------------------------
    def tick(self):
        now = time.monotonic()
        dt, self.last = min(0.1, now - self.last), now
        pedal = 1.0 if self.floored else self.pedal_slider.value
        if self.hold.get():
            self.powertrain.speed = self.hold_slider.value
        else:
            self.powertrain.step(pedal, dt)
        speed = self.powertrain.speed
        rpm = speed * engine.RPM_PER_KMH
        st = voices.State(motor_ratio=max(0.0, min(1.0, rpm / engine.MAX_MOTOR_RPM)),
                          pedal=pedal, speed_kmh=speed, swell=self.swell.update(pedal, dt),
                          active=rpm > engine.MUTE_BELOW_RPM, dt=dt)
        info = self.voice.update(st)
        self.speed_lbl.configure(text=f"{speed:3.0f} km/h")
        self.info_lbl.configure(text=f"pedal {pedal * 100:3.0f} %   {info}")
        if self.floored:
            self.pedal_slider.draw()
        self.root.after(20, self.tick)

    def _flash(self, msg):
        self.status.configure(text=msg)

    def close(self):
        self.graph.stop()
        self.root.destroy()


def _nice_step(span):
    if span <= 0:
        return 0.01
    mag = 10 ** math.floor(math.log10(span / 200))
    return mag


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="Dreadnought")
    ap.add_argument("--null-audio", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    root = tk.Tk()
    Lab(root, args.voice, null_audio=args.null_audio)
    root.mainloop()


if __name__ == "__main__":
    main()
