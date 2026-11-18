#!/usr/bin/env python3
"""Map the i3 recording to speed by ear.

    .venv/bin/python mapper.py                    # the i3 recording
    .venv/bin/python mapper.py some_sound.mp3     # any other audio (or use "Open audio…")

The spectrogram shows the recording (time across, 0-1200 Hz up). Below it are two rows of
markers: SPEEDING UP (green) and SLOWING DOWN (orange). Each marker says "at this many
km/h, play this spot of the file". The speed slider plays the whole voice with your map;
the white line shows where in the file it is reading. Save writes recording_map.json,
which main.py's "Jetsons i3 (recording)" voice loads automatically.

Mouse
  click / drag on the spectrogram     audition that spot (drag to scrub)
  click / drag a marker               select it, move it in time (and hear its spot)
  right-click                         add a marker to the active row
  scroll wheel                        zoom around the pointer
  shift + scroll, middle-drag         pan
Keys
  Delete      remove the selected marker
  Esc         stop auditioning (back to the speed slider)
  F           fit the whole recording in view
  Ctrl+S      save
"""
from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from signalflow import AudioGraph, AudioGraphConfig

import voices

# --- look -------------------------------------------------------------------------------
BG = "#14161b"
PANEL = "#1c1f26"
CARD = "#232731"
LINE = "#2e3340"
TEXT = "#e7e9ee"
MUTED = "#8b92a3"
ACCENT = "#5b8cff"
ROW_COLOR = {"accel": "#3ddc84", "decel": "#ffa24a"}
ROW_NAME = {"accel": "Speeding up", "decel": "Slowing down"}

HINT = ("Click: audition  ·  Drag marker: move  ·  Right-click: add  ·  Scroll: zoom  ·  "
        "Shift+scroll: pan  ·  Del  ·  Esc: stop  ·  F: fit  ·  Ctrl+S: save")
F_MAX = 1200.0          # spectrogram top, Hz
GUTTER = 118            # left label column on the canvas, px
AXIS_H = 26             # time axis height, px
LANE_H = 24             # one lane of marker chips, px
CHIP_W = 36             # marker chip width, px
SPEC_COLS_PER_SEC = 400
SPEC_MAX_COLS = 12000   # long files get a coarser spectrogram instead of a slow one
SPEC_ROWS = 360


def spectrogram(mono, sr, cols, rows):
    """Normalised (0..1) log-magnitude spectrogram, rows from F_MAX down to 0 Hz."""
    n = 4096
    win = np.hanning(n)
    pad = np.concatenate([np.zeros(n // 2), mono, np.zeros(n)])
    hop = len(mono) / cols
    fr = np.fft.rfftfreq(n, 1 / sr)
    bins = np.clip(np.searchsorted(fr, np.linspace(F_MAX, 0, rows)), 0, len(fr) - 1)
    out = np.empty((rows, cols), np.float32)
    for c0 in range(0, cols, 256):
        idx = np.arange(c0, min(cols, c0 + 256))
        frames = np.stack([pad[int(i * hop):int(i * hop) + n] for i in idx]) * win
        mag = np.abs(np.fft.rfft(frames, axis=1))[:, bins]
        out[:, idx] = mag.T
    db = 20 * np.log10(out + 1e-9)
    return np.clip((db - (db.max() - 65)) / 65, 0, 1)


def colormap(img):
    """0..1 -> RGB uint8: deep navy -> violet -> orange -> pale yellow."""
    r = np.clip(img * 2.0 - 0.15, 0, 1)
    g = np.clip(img * 1.7 - 0.65, 0, 1)
    b = np.clip(0.18 + np.where(img < 0.5, img * 1.1, 1.05 - img * 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def nice_step(span_s, width_px, min_px=80):
    for step in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        if step / span_s * width_px >= min_px:
            return step
    return 5.0


class Slider(tk.Canvas):
    """Flat slider: accent-filled track and a round handle, bound to a DoubleVar."""

    def __init__(self, parent, variable, lo, hi, command=None, bg=CARD):
        super().__init__(parent, height=30, bg=bg, highlightthickness=0, cursor="hand2")
        self.var, self.lo, self.hi, self.command = variable, lo, hi, command
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<ButtonPress-1>", self._set)
        self.bind("<B1-Motion>", self._set)
        variable.trace_add("write", lambda *_: self.draw())

    def _x(self, v):
        w = self.winfo_width()
        return 12 + (v - self.lo) / (self.hi - self.lo) * (w - 24)

    def _set(self, ev):
        w = self.winfo_width()
        v = self.lo + (ev.x - 12) / max(1, w - 24) * (self.hi - self.lo)
        self.var.set(round(min(self.hi, max(self.lo, v)) * 2) / 2)
        if self.command:
            self.command()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        y = h / 2
        x = self._x(self.var.get())
        self.create_line(12, y, w - 12, y, fill=LINE, width=6, capstyle="round")
        self.create_line(12, y, x, y, fill=ACCENT, width=6, capstyle="round")
        for k in range(0, int(self.hi) + 1, 10):
            kx = self._x(k)
            self.create_line(kx, y + 7, kx, y + 10, fill=MUTED)
        self.create_oval(x - 9, y - 9, x + 9, y + 9, fill="white", outline=ACCENT, width=2)


class Mapper:
    def __init__(self, root, audio=None, null_audio=False):
        self.root = root
        cfg = AudioGraphConfig()
        if null_audio:
            cfg.backend_name = "null"
        self.graph = AudioGraph(cfg, start=not null_audio)
        self.voice = None
        self._make_voice(audio)
        self.selected = None                      # (row, index)
        self.dragging = None
        self.panning = None
        self.sweep_t = None
        self.dirty = False
        self.chips = []                           # (x0, y0, x1, y1, row, index)
        self.layout = {}
        self.photo = None
        self._redraw_pending = False
        self.last = time.monotonic()

        self._style()
        self._build()
        self._bind()
        self.tick()

    # =====================================================================================
    # audio file
    # =====================================================================================
    def _make_voice(self, audio):
        """Build the voice for an audio file (None = the i3 recording) and analyse it."""
        is_i3 = audio is None or Path(audio).resolve() == Path(voices.RecordingI3.FILE).resolve()
        voice = voices.RecordingI3() if is_i3 else voices.RecordingVoice(audio)
        if self.voice is not None:
            self.voice.output.stop()
        self.voice = voice
        self.voice.set_small_speaker(False)
        self.voice.output.play()
        mono = np.array(self.voice._buf.data).mean(axis=0)
        self.sr = int(self.voice._buf.sample_rate)
        self.duration = len(mono) / self.sr
        cols = int(min(SPEC_MAX_COLS, max(400, self.duration * SPEC_COLS_PER_SEC)))
        self.rgb = colormap(spectrogram(mono, self.sr, cols, SPEC_ROWS))
        self.t0, self.t1 = 0.0, self.duration
        self.dirty = False

    def open_audio(self, path=None):
        if self.dirty and not messagebox.askyesno(
                "Unsaved changes", "Discard the unsaved changes to this map?"):
            return
        path = path or filedialog.askopenfilename(
            title="Open audio", initialdir=str(voices.PROJECT_DIR),
            filetypes=[("Audio", " ".join("*" + e for e in voices.AUDIO_TYPES)),
                       ("All files", "*")])
        if not path:
            return
        self.status.configure(text=f"Loading {Path(path).name} …")
        self.root.update_idletasks()
        try:
            self._make_voice(path)
        except Exception as e:                      # unreadable file: keep the current one
            messagebox.showerror("Couldn't open audio", f"{Path(path).name}:\n{e}")
            self.status.configure(text=HINT)
            return
        self.selected = None
        self.pulse_until.set(f"{self.voice.pulse_until:g}")
        self._update_selection_panel()
        self._update_file_label()
        self._update_pulse_option()
        self.request_redraw()
        mapped = self.voice.map_path.exists()
        self.status.configure(text=f"Opened {Path(path).name} — "
                                   + ("loaded its saved map." if mapped else
                                      "no map yet: starting from a straight ramp, "
                                      "drag the green markers to taste."))

    def _update_pulse_option(self):
        has = bool(self.voice.GROUPS)
        self.pulse_box.state(["!disabled"] if has else ["disabled"])
        self.pulse_note.configure(
            text="One grain per pulse keeps a slow putter crisp. 0 = off." if has else
            "Putter mode needs measured pulse onsets — only available for the i3 recording.")

    # =====================================================================================
    # UI construction
    # =====================================================================================
    def _style(self):
        r = self.root
        r.title("Recording → speed mapper")
        r.configure(bg=BG)
        r.geometry("1360x820")
        r.minsize(1120, 720)
        s = ttk.Style(r)
        s.theme_use("clam")
        base = ("TkDefaultFont", 10)
        s.configure(".", background=BG, foreground=TEXT, fieldbackground=CARD, font=base,
                    bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, troughcolor=CARD,
                    focuscolor=ACCENT, insertcolor=TEXT)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("Card.TFrame", background=CARD)
        s.configure("Header.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Panel.TLabel", background=PANEL)
        s.configure("Card.TLabel", background=CARD)
        s.configure("Muted.Card.TLabel", background=CARD, foreground=MUTED,
                    font=("TkDefaultFont", 9))
        s.configure("CardTitle.TLabel", background=CARD, foreground=MUTED,
                    font=("TkDefaultFont", 9, "bold"))
        s.configure("Title.TLabel", background=PANEL, font=("TkDefaultFont", 14, "bold"))
        s.configure("Sub.TLabel", background=PANEL, foreground=MUTED)
        s.configure("Big.Card.TLabel", background=CARD, font=("TkDefaultFont", 26, "bold"))
        s.configure("Status.TLabel", background=PANEL, foreground=MUTED, padding=(12, 6))
        s.configure("TButton", background=CARD, foreground=TEXT, padding=(12, 6),
                    borderwidth=1, relief="flat")
        s.map("TButton", background=[("active", LINE), ("disabled", PANEL)],
              foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=ACCENT, foreground="white")
        s.map("Accent.TButton", background=[("active", "#7aa2ff")])
        s.configure("Danger.TButton", background="#3a2328", foreground="#ff8a8a")
        s.map("Danger.TButton", background=[("active", "#52282f")])
        s.configure("TRadiobutton", background=CARD, foreground=TEXT, padding=(2, 3))
        s.map("TRadiobutton", background=[("active", CARD)])
        s.configure("TSpinbox", arrowsize=13, padding=(6, 4), foreground=TEXT,
                    background=CARD, arrowcolor=TEXT)
        s.configure("Horizontal.TScale", background=ACCENT, troughcolor=LINE,
                    sliderthickness=18)

    def _card(self, parent, title):
        outer = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12))
        outer.pack(fill="x", pady=(0, 10))
        ttk.Label(outer, text=title.upper(), style="CardTitle.TLabel").pack(anchor="w",
                                                                         pady=(0, 8))
        return outer

    def _build(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(1, weight=1)

        # header ------------------------------------------------------------------------
        head = ttk.Frame(r, style="Header.TFrame", padding=(16, 10))
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        head.columnconfigure(1, weight=1)
        ttk.Label(head, text="Recording → speed mapper", style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        self.file_label = ttk.Label(head, text="", style="Sub.TLabel")
        self.file_label.grid(row=0, column=1, sticky="w", padx=14)
        btns = ttk.Frame(head, style="Header.TFrame")
        btns.grid(row=0, column=2, sticky="e")
        ttk.Button(btns, text="Open audio…", command=self.open_audio).pack(side="left",
                                                                       padx=4)
        ttk.Button(btns, text="Fit view", command=self.fit).pack(side="left", padx=4)
        ttk.Button(btns, text="Revert to defaults", command=self.defaults).pack(side="left",
                                                                                padx=4)
        self.save_btn = ttk.Button(btns, text="Save", style="Accent.TButton",
                                   command=self.save)
        self.save_btn.pack(side="left", padx=(4, 0))

        # canvas ------------------------------------------------------------------------
        body = ttk.Frame(r, padding=(12, 12, 6, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, bg=PANEL, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # playback bar under the view -----------------------------------------------------
        play = ttk.Frame(body, style="Card.TFrame", padding=(16, 10))
        play.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        play.columnconfigure(1, weight=1)
        top = ttk.Frame(play, style="Card.TFrame")
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="PLAYBACK", style="CardTitle.TLabel").grid(row=0, column=0,
                                                                       sticky="w")
        self.where = ttk.Label(top, text="", style="Muted.Card.TLabel")
        self.where.grid(row=0, column=1, sticky="w", padx=(14, 8))
        ttk.Button(top, text="▶  Sweep 0 → top → 0", command=self.sweep).grid(
            row=0, column=2, padx=(0, 6))
        ttk.Button(top, text="■  Stop", command=self.stop_all).grid(row=0, column=3)
        speed_box = ttk.Frame(play, style="Card.TFrame")
        speed_box.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.speed_label = ttk.Label(speed_box, text="0", style="Big.Card.TLabel", width=3,
                                     anchor="e")
        self.speed_label.pack(side="left")
        ttk.Label(speed_box, text="km/h", style="Muted.Card.TLabel").pack(
            side="left", anchor="s", pady=(0, 9), padx=(4, 12))
        self.dir_badge = tk.Label(speed_box, text="STOPPED", bg=LINE, fg=MUTED, padx=10,
                                  pady=3, width=11, font=("TkDefaultFont", 8, "bold"))
        self.dir_badge.pack(side="left")
        self.speed = tk.DoubleVar(value=0.0)
        Slider(play, self.speed, 0, 80, command=self.stop_audition).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=(4, 0))

        # side panel --------------------------------------------------------------------
        side = ttk.Frame(r, padding=(6, 12, 12, 12))
        side.grid(row=1, column=1, sticky="ns")
        side.configure(width=300)

        c = self._card(side, "Selected marker")
        self.sel_title = ttk.Label(c, text="Nothing selected", style="Card.TLabel",
                                   font=("TkDefaultFont", 11, "bold"))
        self.sel_title.pack(anchor="w")
        grid = ttk.Frame(c, style="Card.TFrame")
        grid.pack(fill="x", pady=(8, 0))
        grid.columnconfigure(1, weight=1)
        self.sel_speed = tk.StringVar()
        self.sel_time = tk.StringVar()
        for i, (label, var, inc, cmd) in enumerate((
                ("Speed (km/h)", self.sel_speed, 0.5, self.apply_speed),
                ("Time in file (s)", self.sel_time, 0.01, self.apply_time))):
            ttk.Label(grid, text=label, style="Card.TLabel").grid(row=i, column=0, sticky="w",
                                                                  pady=3)
            sb = ttk.Spinbox(grid, from_=0, to=300, increment=inc, width=8, textvariable=var,
                             command=cmd)
            sb.grid(row=i, column=1, sticky="e", pady=3)
            sb.bind("<Return>", lambda e, f=cmd: f())
            sb.bind("<FocusOut>", lambda e, f=cmd: f())
        self.del_btn = ttk.Button(c, text="Delete marker", style="Danger.TButton",
                                  command=self.delete_marker)
        self.del_btn.pack(fill="x", pady=(10, 0))

        c = self._card(side, "Options")
        ttk.Label(c, text="New markers go to", style="Card.TLabel").pack(anchor="w")
        self.row = tk.StringVar(value="accel")
        radios = ttk.Frame(c, style="Card.TFrame")
        radios.pack(fill="x", pady=(2, 8))
        for key in ("accel", "decel"):
            ttk.Radiobutton(radios, text=ROW_NAME[key], value=key,
                            variable=self.row).pack(side="left", padx=(0, 10))
        o = ttk.Frame(c, style="Card.TFrame")
        o.pack(fill="x")
        o.columnconfigure(0, weight=1)
        ttk.Label(o, text="Putter mode until (km/h)", style="Card.TLabel").grid(
            row=0, column=0, sticky="w")
        self.pulse_until = tk.StringVar(value=f"{self.voice.pulse_until:g}")
        pb = ttk.Spinbox(o, from_=0, to=20, increment=0.5, width=6,
                         textvariable=self.pulse_until, command=self.apply_pulse)
        pb.grid(row=0, column=1, sticky="e")
        pb.bind("<Return>", lambda e: self.apply_pulse())
        self.pulse_box = pb
        self.pulse_note = ttk.Label(c, text="", style="Muted.Card.TLabel", wraplength=250)
        self.pulse_note.pack(anchor="w", pady=(4, 0))

        # status bar --------------------------------------------------------------------
        self.status = ttk.Label(r, text=HINT, style="Status.TLabel", anchor="w")
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._update_selection_panel()
        self._update_file_label()
        self._update_pulse_option()

    def _bind(self):
        c, r = self.canvas, self.root
        c.bind("<Configure>", lambda e: self.request_redraw())
        c.bind("<ButtonPress-1>", self.press)
        c.bind("<B1-Motion>", self.drag)
        c.bind("<ButtonRelease-1>", self.release)
        c.bind("<ButtonPress-3>", self.add_marker)
        c.bind("<ButtonPress-2>", self.pan_start)
        c.bind("<B2-Motion>", self.pan_move)
        c.bind("<MouseWheel>", self.wheel)                       # Windows / macOS
        c.bind("<Button-4>", lambda e: self.wheel(e, +1))        # X11
        c.bind("<Button-5>", lambda e: self.wheel(e, -1))
        c.bind("<Shift-Button-4>", lambda e: self.pan_by(-0.1))
        c.bind("<Shift-Button-5>", lambda e: self.pan_by(+0.1))
        r.bind("<Delete>", lambda e: self._key_delete())
        r.bind("<Escape>", lambda e: self.stop_all())
        r.bind("<Control-s>", lambda e: self.save())
        r.bind("<Control-o>", lambda e: self.open_audio())
        r.bind("f", lambda e: None if self._typing() else self.fit())

    def _typing(self):
        return isinstance(self.root.focus_get(), (ttk.Spinbox, tk.Entry, ttk.Entry))

    def _key_delete(self):
        if not self._typing():
            self.delete_marker()

    # =====================================================================================
    # geometry
    # =====================================================================================
    def lanes(self):
        """Greedy lane assignment so chips never overlap; returns {row: [lane per index]}."""
        out = {}
        for row in ("accel", "decel"):
            pts = self.points(row)
            assign, lane_end = [0] * len(pts), []
            for i in sorted(range(len(pts)), key=lambda i: pts[i][1]):
                x = self.x_of(pts[i][1])
                for lane, end in enumerate(lane_end):
                    if x - CHIP_W / 2 > end + 4:
                        lane_end[lane] = x + CHIP_W / 2
                        assign[i] = lane
                        break
                else:
                    lane_end.append(x + CHIP_W / 2)
                    assign[i] = len(lane_end) - 1
            out[row] = assign
        return out

    def _compute_layout(self):
        w = max(200, self.canvas.winfo_width())
        h = max(200, self.canvas.winfo_height())
        self.layout["plot_w"] = w - GUTTER - 14
        lanes = self.lanes()
        n_lanes = {row: max([1] + [l + 1 for l in lanes[row]]) for row in lanes}
        rows_h = sum(n_lanes.values()) * LANE_H + 2 * 10
        spec_h = max(120, h - rows_h - AXIS_H - 12)
        self.layout.update(w=w, h=h, spec_y0=8, spec_h=spec_h, lanes=lanes, n_lanes=n_lanes)
        y = 8 + spec_h + AXIS_H
        self.layout["row_y"] = {}
        for row in ("accel", "decel"):
            self.layout["row_y"][row] = y
            y += n_lanes[row] * LANE_H + 10

    def x_of(self, t):
        return GUTTER + (t - self.t0) / (self.t1 - self.t0) * self.layout.get("plot_w", 800)

    def t_of(self, x):
        t = self.t0 + (x - GUTTER) / self.layout.get("plot_w", 800) * (self.t1 - self.t0)
        return min(self.duration, max(0.0, t))

    def points(self, row):
        return self.voice.accel if row == "accel" else self.voice.decel

    def in_spec(self, x, y):
        L = self.layout
        return GUTTER <= x <= GUTTER + L["plot_w"] and L["spec_y0"] <= y <= L["spec_y0"] + L["spec_h"]

    # =====================================================================================
    # drawing
    # =====================================================================================
    def request_redraw(self, image=True):
        self._need_image = getattr(self, "_need_image", False) or image
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after_idle(self._redraw)

    def _redraw(self):
        self._redraw_pending = False
        self._compute_layout()
        if self._need_image:
            self._draw_image()
            self._need_image = False
        self._draw_overlay()

    def _draw_image(self):
        L = self.layout
        pw, ph = int(L["plot_w"]), int(L["spec_h"])
        cols = self.rgb.shape[1]
        c0 = self.t0 / self.duration * cols
        c1 = self.t1 / self.duration * cols
        ci = np.clip(np.linspace(c0, c1, pw, endpoint=False).astype(int), 0, cols - 1)
        ri = np.clip(np.linspace(0, SPEC_ROWS, ph, endpoint=False).astype(int), 0, SPEC_ROWS - 1)
        img = self.rgb[ri][:, ci]
        ppm = b"P6 %d %d 255\n" % (pw, ph) + np.ascontiguousarray(img).tobytes()
        self.photo = tk.PhotoImage(data=ppm, format="PPM")
        self.canvas.delete("image")
        self.canvas.create_image(GUTTER, L["spec_y0"], image=self.photo, anchor="nw",
                                 tags="image")
        self.canvas.tag_lower("image")

    def _draw_overlay(self):
        c, L = self.canvas, self.layout
        c.delete("overlay")
        x0, x1 = GUTTER, GUTTER + L["plot_w"]
        sy0, sy1 = L["spec_y0"], L["spec_y0"] + L["spec_h"]

        # frequency labels
        for f in (100, 300, 500, 700, 900, 1100):
            y = sy0 + L["spec_h"] * (1 - f / F_MAX)
            c.create_text(GUTTER - 8, y, text=f"{f} Hz", anchor="e", fill=MUTED,
                          font=("TkDefaultFont", 8), tags="overlay")

        # time axis
        span = self.t1 - self.t0
        step = nice_step(span, L["plot_w"])
        t = np.ceil(self.t0 / step) * step
        while t <= self.t1 + 1e-9:
            x = self.x_of(t)
            c.create_line(x, sy1, x, sy1 + 6, fill=MUTED, tags="overlay")
            label = f"{t:.2f}".rstrip("0").rstrip(".") + " s"
            c.create_text(x, sy1 + 8, text=label, anchor="n", fill=MUTED,
                          font=("TkDefaultFont", 8), tags="overlay")
            c.create_line(x, L["row_y"]["accel"] - 4, x, L["h"], fill="#20242c", tags="overlay")
            t += step

        # marker rows
        for row in ("accel", "decel"):
            y0 = L["row_y"][row]
            hgt = L["n_lanes"][row] * LANE_H
            col = ROW_COLOR[row]
            c.create_rectangle(x0, y0 - 4, x1, y0 + hgt + 4, fill=BG, outline="",
                               tags="overlay")
            c.create_rectangle(4, y0 - 4, GUTTER - 10, y0 + hgt + 4, fill=BG, outline="",
                               tags="overlay")
            c.create_rectangle(4, y0 - 4, 8, y0 + hgt + 4, fill=col, outline="", tags="overlay")
            c.create_text(16, y0 + hgt / 2, text=ROW_NAME[row], anchor="w", fill=col,
                          font=("TkDefaultFont", 9, "bold"), tags="overlay")

        self.chips = []
        for row in ("accel", "decel"):
            col = ROW_COLOR[row]
            for i, (kmh, t) in enumerate(self.points(row)):
                if not (self.t0 - 0.5 <= t <= self.t1 + 0.5):
                    continue
                x = self.x_of(t)
                sel = self.selected == (row, i)
                lane = L["lanes"][row][i]
                cy0 = L["row_y"][row] + lane * LANE_H + 2
                cy1 = cy0 + LANE_H - 4
                if x0 <= x <= x1:
                    # guide line on the spectrogram; the selected one solid and bold
                    c.create_line(x, sy0, x, sy1, fill=col, dash=() if sel else (2, 4),
                                  width=3 if sel else 1, tags="overlay")
                    # leader stays inside its own row so it never crosses the time axis
                    c.create_line(x, L["row_y"][row] - 4, x, cy0, fill=col, tags="overlay")
                fill = col if sel else CARD
                fg = BG if sel else col
                c.create_rectangle(x - CHIP_W / 2, cy0, x + CHIP_W / 2, cy1, fill=fill,
                                   outline=col, width=1, tags="overlay")
                c.create_text(x, (cy0 + cy1) / 2, text=f"{kmh:g}", fill=fg,
                              font=("TkDefaultFont", 9, "bold"), tags="overlay")
                self.chips.append((x - CHIP_W / 2, cy0, x + CHIP_W / 2, cy1, row, i))

        # zoom indicator
        if self.t1 - self.t0 < self.duration - 1e-6:
            frac0, frac1 = self.t0 / self.duration, self.t1 / self.duration
            c.create_rectangle(x0, L["h"] - 5, x1, L["h"] - 2, fill=LINE, outline="",
                               tags="overlay")
            c.create_rectangle(x0 + frac0 * L["plot_w"], L["h"] - 5,
                               x0 + frac1 * L["plot_w"], L["h"] - 2, fill=MUTED, outline="",
                               tags="overlay")

        self.cursor = c.create_line(0, sy0, 0, sy1, fill="white", width=2, tags="overlay")
        self._move_cursor()

    def _move_cursor(self):
        L = self.layout
        if not L:
            return
        x = self.x_of(self.voice.cursor)
        visible = GUTTER <= x <= GUTTER + L["plot_w"]
        self.canvas.coords(self.cursor, x, L["spec_y0"], x, L["spec_y0"] + L["spec_h"])
        self.canvas.itemconfigure(self.cursor, state="normal" if visible else "hidden")

    # =====================================================================================
    # view: zoom / pan
    # =====================================================================================
    def set_view(self, t0, t1):
        span = min(self.duration, max(0.25, t1 - t0))
        t0 = min(max(0.0, t0), self.duration - span)
        self.t0, self.t1 = t0, t0 + span
        self.request_redraw()

    def fit(self):
        self.set_view(0.0, self.duration)

    def wheel(self, ev, direction=None):
        if direction is None:
            direction = 1 if ev.delta > 0 else -1
        if ev.state & 0x1:                               # shift: pan
            self.pan_by(-0.1 * direction)
            return
        anchor = self.t_of(ev.x)
        factor = 0.8 if direction > 0 else 1.25
        self.set_view(anchor - (anchor - self.t0) * factor, anchor + (self.t1 - anchor) * factor)

    def pan_by(self, frac):
        span = self.t1 - self.t0
        self.set_view(self.t0 + frac * span, self.t1 + frac * span)

    def pan_start(self, ev):
        self.panning = (ev.x, self.t0, self.t1)

    def pan_move(self, ev):
        if self.panning:
            x, t0, t1 = self.panning
            dt = (x - ev.x) / self.layout["plot_w"] * (t1 - t0)
            self.set_view(t0 + dt, t1 + dt)

    # =====================================================================================
    # markers
    # =====================================================================================
    def hit(self, x, y):
        for cx0, cy0, cx1, cy1, row, i in reversed(self.chips):
            if cx0 <= x <= cx1 and cy0 <= y <= cy1:
                return row, i
        if self.in_spec(x, y):                           # near a marker's guide line
            best = None
            for row in ("accel", "decel"):
                for i, (_, t) in enumerate(self.points(row)):
                    d = abs(self.x_of(t) - x)
                    if d <= 4 and (best is None or d < best[0]):
                        best = (d, row, i)
            if best:
                return best[1], best[2]
        return None

    def select(self, sel):
        self.selected = sel
        self._update_selection_panel()
        self.request_redraw(image=False)

    def _update_selection_panel(self):
        if self.selected:
            row, i = self.selected
            kmh, t = self.points(row)[i]
            self.sel_title.configure(text=f"{ROW_NAME[row]}", foreground=ROW_COLOR[row])
            self.sel_speed.set(f"{kmh:g}")
            self.sel_time.set(f"{t:.3f}")
            self.del_btn.state(["!disabled"])
        else:
            self.sel_title.configure(text="Nothing selected", foreground=MUTED)
            self.sel_speed.set("")
            self.sel_time.set("")
            self.del_btn.state(["disabled"])

    def commit(self, row, pts):
        if row == "accel":
            self.voice.set_map(pts, self.voice.decel)
        else:
            self.voice.set_map(self.voice.accel, pts)
        self.dirty = True
        self._update_file_label()

    def _replace_selected(self, kmh=None, t=None):
        row, i = self.selected
        pts = list(self.points(row))
        k0, t0 = pts[i]
        new = (k0 if kmh is None else kmh, t0 if t is None else t)
        pts[i] = new
        self.commit(row, pts)
        self.selected = (row, self.points(row).index(new))
        return new

    def apply_speed(self):
        if not self.selected:
            return
        try:
            kmh = max(0.0, float(self.sel_speed.get()))
        except ValueError:
            return
        self._replace_selected(kmh=kmh)
        self.select(self.selected)

    def apply_time(self):
        if not self.selected:
            return
        try:
            t = min(self.duration, max(0.0, float(self.sel_time.get())))
        except ValueError:
            return
        self._replace_selected(t=t)
        self.audition_t(t)
        self.select(self.selected)

    def apply_pulse(self):
        try:
            v = max(0.0, float(self.pulse_until.get()))
        except ValueError:
            return
        self.voice.set_map(self.voice.accel, self.voice.decel, pulse_until=v,
                           hum_from=v + 1.5)
        self.dirty = True
        self._update_file_label()

    def add_marker(self, ev):
        row = self.row.get()
        t = self.t_of(ev.x)
        by_t = sorted(self.points(row), key=lambda p: p[1])
        kmh = 0.0
        if by_t:
            before = [p for p in by_t if p[1] <= t]
            after = [p for p in by_t if p[1] > t]
            if before and after:
                (k0, t0), (k1, t1) = before[-1], after[0]
                kmh = k0 + (k1 - k0) * (t - t0) / (t1 - t0)
            else:
                kmh = (before[-1] if before else after[0])[0]
        kmh = round(kmh * 2) / 2
        pts = list(self.points(row)) + [(kmh, t)]
        self.commit(row, pts)
        self.select((row, self.points(row).index((kmh, t))))
        self.audition_t(t)
        self.status.configure(text=f"Added a {ROW_NAME[row].lower()} marker at {t:.2f} s "
                                   f"({kmh:g} km/h) — set its speed in the panel")

    def delete_marker(self, _ev=None):
        if not self.selected:
            return
        row, i = self.selected
        pts = list(self.points(row))
        if row == "accel" and len(pts) <= 2:
            self.status.configure(text="The speeding-up row needs at least two markers.")
            return
        pts.pop(i)
        self.commit(row, pts)
        self.select(None)

    # =====================================================================================
    # mouse
    # =====================================================================================
    def press(self, ev):
        self.canvas.focus_set()
        h = self.hit(ev.x, ev.y)
        if h:
            self.select(h)
            self.dragging = h
            self.audition_t(self.points(h[0])[h[1]][1])
        elif self.in_spec(ev.x, ev.y):
            self.audition(ev.x)

    def drag(self, ev):
        if self.dragging:
            self.selected = self.dragging
            new = self._replace_selected(t=self.t_of(ev.x))
            self.dragging = self.selected
            self._update_selection_panel()
            self.audition_t(new[1])
            self.request_redraw(image=False)
        elif self.in_spec(ev.x, ev.y):
            self.audition(ev.x)

    def release(self, _ev):
        self.dragging = None

    def audition(self, x):
        self.audition_t(self.t_of(x))

    def audition_t(self, t):
        self.sweep_t = None
        self.voice.audition = t

    def stop_audition(self):
        self.voice.audition = None

    def stop_all(self):
        self.stop_audition()
        self.sweep_t = None
        self.speed.set(0.0)

    # =====================================================================================
    # playback loop
    # =====================================================================================
    def sweep(self):
        self.stop_audition()
        self.sweep_t = 0.0

    def tick(self):
        now = time.monotonic()
        dt, self.last = now - self.last, now
        if self.sweep_t is not None:
            top = max(10.0, self.voice.top_kmh)
            up, hold, down = 8.0, 1.0, 6.0
            s = self.sweep_t
            if s < up:
                v = top * s / up
            elif s < up + hold:
                v = top
            elif s < up + hold + down:
                v = top * (1 - (s - up - hold) / down)
            else:
                v, self.sweep_t = 0.0, None
            self.speed.set(round(v, 1))
            if self.sweep_t is not None:
                self.sweep_t += dt
        speed = self.speed.get()
        info = self.voice.update(voices.State(
            motor_ratio=min(1.0, speed / 80.0), pedal=0.3, speed_kmh=speed, swell=0.0,
            active=speed > 0.4, dt=dt))

        self.speed_label.configure(text=f"{speed:.0f}")
        if self.voice.audition is not None:
            badge, bg, fg = "AUDITION", ACCENT, "white"
            self.where.configure(text=f"Holding {self.voice.audition:.2f} s of the recording")
            self.status.configure(text="Auditioning — Esc or move the speed slider to go back")
        elif speed <= 0.4:
            badge, bg, fg = "STOPPED", LINE, MUTED
            self.where.configure(text="Silent at a standstill")
            if self.status.cget("text").startswith("Auditioning"):
                self.status.configure(text=HINT)
        else:
            slowing = info.startswith("slowing")
            badge = "SLOWING ↓" if slowing else "SPEEDING ↑"
            bg = ROW_COLOR["decel" if slowing else "accel"]
            fg = BG
            self.where.configure(text=f"Reading {self.voice.cursor:.2f} s of the recording")
            if self.status.cget("text").startswith("Auditioning"):
                self.status.configure(text=HINT)
        self.dir_badge.configure(text=badge, bg=bg, fg=fg)
        if self.layout:
            self._move_cursor()
        self.root.after(20, self.tick)

    # =====================================================================================
    # file
    # =====================================================================================
    def _update_file_label(self):
        v = self.voice
        mp = v.map_path if v.map_path.exists() else getattr(v, "LEGACY_MAP", None)
        src = f"maps/{v.map_path.name}" if v.map_path.exists() else (
            mp.name if mp is not None and mp.exists() else "not saved yet")
        self.file_label.configure(
            text=f"{v.source.name}   ·   {src}{'   ·   unsaved changes' if self.dirty else ''}")
        self.save_btn.configure(text="Save •" if self.dirty else "Save")
        self.root.title(("• " if self.dirty else "") + "Recording → speed mapper")

    def save(self):
        path = self.voice.save_map()
        self.dirty = False
        self._update_file_label()
        self.status.configure(text=f"Saved maps/{path.name} — main.py lists it as "
                                   f"\"{self.voice.name}\" (restart main.py to pick it up).")

    def defaults(self):
        v = self.voice
        v.set_map(*v.default_map())
        self.pulse_until.set(f"{v.pulse_until:g}")
        self.dirty = True
        self._update_file_label()
        self.select(None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="audio file to map (default: the i3 recording)")
    ap.add_argument("--null-audio", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    root = tk.Tk()
    Mapper(root, audio=args.audio, null_audio=args.null_audio)
    root.mainloop()


if __name__ == "__main__":
    main()
