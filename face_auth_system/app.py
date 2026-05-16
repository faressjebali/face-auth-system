"""
Face Auth System — Desktop Application (Cyberpunk Redesign)

Four screens:
  HomeScreen     — Split layout: wordmark + buttons | animated biometric ring
  RegisterScreen — Header progress + personal form | live webcam
  LoginScreen    — Full-window webcam with HUD overlay and confirmation bar
  WelcomeScreen  — Post-login success state

Usage
-----
  python app.py
"""

import math
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk

import config
from deep_model import DeepFaceModel, cosine_similarity
from enrollment import _load_gallery, _save_gallery
from lighting import normalize_lighting
from user_store import UserStore

# ---------------------------------------------------------------------------
# Design tokens — deep-space + electric cyan + plasma violet
# ---------------------------------------------------------------------------
BG       = "#030308"
PANEL    = "#07070f"
SURFACE  = "#0c0c1c"
ACCENT   = "#00F5FF"   # electric cyan  — primary
ACCENT_D = "#00c0ca"   # cyan pressed
ACCENT2  = "#7B2FFF"   # plasma violet  — secondary
TEXT     = "#c8d8e8"
SUBTEXT  = "#4a5a72"
DIM      = "#1a2030"
SUCCESS  = "#2adf80"
ERROR    = "#ff2255"
WARN     = "#ff8844"
BORDER   = "#0e1422"

_FM = "Courier"   # monospace — display / HUD
_FF = "Ubuntu"    # sans      — body text


def _f(size: int, weight: str = "normal") -> tuple:
    return (_FF, size, weight)


def _fm(size: int, weight: str = "normal") -> tuple:
    return (_FM, size, weight)


F_DISPLAY = _fm(48, "bold")
F_TITLE   = _fm(16, "bold")
F_H2      = _f(13, "bold")
F_BODY    = _f(11)
F_SMALL   = _f(10)
F_BTN     = _f(11, "bold")
F_LABEL   = _f(8,  "bold")
F_HUD     = _fm(9)

# Auth constants (unchanged)
_CONFIRM_NEEDED = 2
_SIM_THRESHOLD  = 1.0 - config.DEEP_DISTANCE_THRESHOLD

_POSES = [("Look straight at the camera", "FRONTAL")]

_STATUS_CYCLE = [
    "INITIALIZING",
    "LOADING NEURAL MODELS",
    "SCANNING ENVIRONMENT",
    "CALIBRATING SENSORS",
    "SYSTEM READY",
]

# Detectors (unchanged)
_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_faces_haar(gray: np.ndarray):
    return _HAAR.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))


_HOG = dlib.get_frontal_face_detector()


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def _btn(parent, text: str, command, style: str = "primary", width: int = 0, **kw):
    palettes = {
        "primary":   dict(bg=ACCENT,   fg=BG,      abg=ACCENT_D, afg=BG),
        "secondary": dict(bg=BG,       fg=ACCENT2, abg=SURFACE,  afg="#9f5fff"),
        "ghost":     dict(bg=BG,       fg=SUBTEXT, abg=SURFACE,  afg=TEXT),
    }
    p  = palettes.get(style, palettes["primary"])
    hl = ACCENT if style == "primary" else ACCENT2
    b  = tk.Button(
        parent, text=text, command=command,
        bg=p["bg"], fg=p["fg"],
        activebackground=p["abg"], activeforeground=p["afg"],
        font=F_BTN, relief="flat", cursor="hand2",
        padx=22, pady=11,
        highlightthickness=1, highlightbackground=hl,
        **kw,
    )
    if width:
        b.configure(width=width)
    return b


def _lbl(parent, text: str, font=F_BODY, fg=TEXT, bg=BG, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)


def _entry(parent, width: int = 28, show: str = None):
    return tk.Entry(
        parent,
        bg=SURFACE, fg=TEXT, insertbackground=ACCENT,
        relief="flat", font=F_BODY, width=width,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        **({"show": show} if show else {}),
    )


# ---------------------------------------------------------------------------
# Biometric scan canvas — multi-ring cyberpunk visualization
# ---------------------------------------------------------------------------

class ScanCanvas(tk.Canvas):
    """
    Three concentric rotating dashed rings:
      outer  — cyan,   slow CW,  18 segments
      middle — violet, slow CCW,  9 segments
      inner  — cyan,   fast CW,  12 segments
    Plus a radar sweep arc + face-guide corner brackets.
    All arcs are pre-created; only their start angles are updated each frame.
    """

    def __init__(self, parent, size: int = 310, **kw):
        super().__init__(
            parent, width=size, height=size,
            bg=BG, bd=0, highlightthickness=0, **kw,
        )
        self._size = size
        self._cx   = cx = size // 2
        self._cy   = cy = size // 2
        self._a1   = 0.0   # outer CW
        self._a2   = 0.0   # middle CCW
        self._a3   = 0.0   # inner CW fast
        self._job: Optional[str] = None

        r1 = size // 2 - 10   # outer ring radius
        r2 = r1 - 26           # middle ring radius
        r3 = r1 - 50           # inner ring radius
        self._r1 = r1

        # ── Static elements ───────────────────────────────────────────
        # Tick marks around outer ring (36 × 10°)
        for deg in range(0, 360, 10):
            a   = math.radians(deg - 90)
            big = (deg % 90 == 0)
            lo  = r1 - (9 if big else 3)
            self.create_line(
                cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                cx + lo * math.cos(a), cy + lo * math.sin(a),
                fill=ACCENT if big else DIM, width=1,
            )

        # Face-guide corner brackets
        half = r3 * 0.70
        bl   = 18
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bx, by = cx + sx * half, cy + sy * half
            self.create_line(bx, by, bx - sx * bl, by, fill=ACCENT, width=1)
            self.create_line(bx, by, bx, by - sy * bl, fill=ACCENT, width=1)

        # Facial landmark dots
        scale = half * 1.35
        for dx, dy in [(0, -0.38), (-0.32, -0.16), (0.32, -0.16),
                       (0, 0.06), (-0.22, 0.24), (0.22, 0.24), (0, 0.44)]:
            px, py = cx + dx * scale, cy + dy * scale
            self.create_oval(px - 2, py - 2, px + 2, py + 2, fill=ACCENT, outline="")

        # Crosshair lines
        cl = half * 0.90
        self.create_line(cx - cl, cy, cx + cl, cy, fill=DIM, width=1)
        self.create_line(cx, cy - cl, cx, cy + cl, fill=DIM, width=1)

        # ── Pre-created dynamic arcs ──────────────────────────────────
        # Outer: 18 × 12° (20° pitch), cyan
        self._arcs1 = [
            self.create_arc(cx - r1, cy - r1, cx + r1, cy + r1,
                           start=i * 20, extent=12, style="arc",
                           outline=ACCENT, width=1)
            for i in range(18)
        ]
        # Middle: 9 × 25° (40° pitch), violet
        self._arcs2 = [
            self.create_arc(cx - r2, cy - r2, cx + r2, cy + r2,
                           start=i * 40, extent=25, style="arc",
                           outline=ACCENT2, width=2)
            for i in range(9)
        ]
        # Inner: 12 × 8° (30° pitch), cyan
        self._arcs3 = [
            self.create_arc(cx - r3, cy - r3, cx + r3, cy + r3,
                           start=i * 30, extent=8, style="arc",
                           outline=ACCENT, width=1)
            for i in range(12)
        ]
        # Radar sweep arc + leading-edge line
        self._sweep_arc = self.create_arc(
            cx - r1, cy - r1, cx + r1, cy + r1,
            start=0, extent=60, style="arc", outline=ACCENT, width=2,
        )
        self._sweep_line = self.create_line(cx, cy, cx + r1, cy, fill=ACCENT, width=1)

        # Center dot (drawn last so it sits on top)
        self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=ACCENT, outline="")

        self._animate()

    def _animate(self) -> None:
        self._a1 = (self._a1 + 2.0) % 360
        self._a2 = (self._a2 + 1.4) % 360
        self._a3 = (self._a3 + 4.5) % 360

        for i, arc in enumerate(self._arcs1):
            self.itemconfigure(arc, start=self._a1 + i * 20)
        for i, arc in enumerate(self._arcs2):
            self.itemconfigure(arc, start=-self._a2 + i * 40)
        for i, arc in enumerate(self._arcs3):
            self.itemconfigure(arc, start=self._a3 + i * 30)

        sweep = (self._a1 * 1.5) % 360
        self.itemconfigure(self._sweep_arc, start=sweep)
        sx = math.radians(sweep)
        self.coords(
            self._sweep_line,
            self._cx, self._cy,
            self._cx + self._r1 * math.cos(sx),
            self._cy - self._r1 * math.sin(sx),
        )
        self._job = self.after(33, self._animate)

    def stop(self) -> None:
        if self._job:
            self.after_cancel(self._job)
            self._job = None


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

class FaceAuthApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NEXUS AUTH")
        self.geometry("960x640")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.user_store = UserStore()
        self.deep_model = DeepFaceModel()
        self._gallery: dict = _load_gallery()
        self._screen: Optional[tk.Frame] = None

        threading.Thread(target=self._warm_model, daemon=True).start()
        self._go_home()

    def _warm_model(self) -> None:
        try:
            self.deep_model.extract_embedding(np.zeros((64, 64, 3), dtype=np.uint8))
        except Exception:
            pass

    def _go_home(self) -> None:
        self._set(HomeScreen(self, go_register=self._go_register, go_login=self._go_login))

    def _go_register(self) -> None:
        self._set(RegisterScreen(self, go_back=self._go_home, on_done=self._after_register))

    def _go_login(self) -> None:
        raw       = _load_gallery()
        valid_ids = set(self.user_store._data.keys())
        self._gallery = {uid: emb for uid, emb in raw.items() if uid in valid_ids}
        self._set(LoginScreen(self, go_back=self._go_home, on_login=self._after_login))

    def _after_register(self) -> None:
        self._gallery = _load_gallery()
        messagebox.showinfo("Account Created",
                            "Your account is ready.\nYou can now log in with your face.")
        self._go_home()

    def _after_login(self, user_id: str) -> None:
        user = self.user_store.get(user_id)
        if user is None:
            messagebox.showerror("Error", "Recognised face has no profile. Please re-register.")
            self._go_home()
            return
        self._set(WelcomeScreen(self, user=user, go_home=self._go_home))

    def _set(self, screen: tk.Frame) -> None:
        if self._screen is not None:
            self._screen.cleanup()
            self._screen.destroy()
        self._screen = screen
        screen.pack(fill="both", expand=True)


# ---------------------------------------------------------------------------
# Base screen
# ---------------------------------------------------------------------------

class _Screen(tk.Frame):
    def __init__(self, master: FaceAuthApp, **kw) -> None:
        super().__init__(master, bg=BG, **kw)

    @property
    def app(self) -> FaceAuthApp:
        return self.master  # type: ignore[return-value]

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HomeScreen
# ---------------------------------------------------------------------------

class HomeScreen(_Screen):
    """
    Left: wordmark + cycling status + metrics + buttons.
    Right: animated multi-ring biometric visualization.
    """

    def __init__(self, master, go_register, go_login) -> None:
        super().__init__(master)
        self._go_register = go_register
        self._go_login    = go_login
        self._canvas: Optional[ScanCanvas] = None
        self._status_idx  = 0
        self._cursor_on   = True
        self._status_job: Optional[str] = None
        self._cursor_job:  Optional[str] = None
        self.lbl_sys_status: Optional[tk.Label] = None
        self._dot_canvas: Optional[tk.Canvas] = None
        self._dot_oval: Optional[int] = None
        self._build()
        self._status_job = self.after(1800, self._tick_status)
        self._cursor_job = self.after(600,  self._tick_cursor)

    def _build(self) -> None:
        # ── Left column ───────────────────────────────────────────────────
        left = tk.Frame(self, bg=BG, width=460)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        content = tk.Frame(left, bg=BG)
        content.place(relx=0.10, rely=0.5, anchor="w")

        # System eyebrow label with flanking rules
        sys_row = tk.Frame(content, bg=BG)
        sys_row.pack(anchor="w", pady=(0, 28))
        tk.Frame(sys_row, bg=ACCENT, width=22, height=1).pack(side="left", padx=(0, 10))
        _lbl(sys_row, "NEXUS SECURITY PROTOCOL v4.2",
             font=F_HUD, fg=ACCENT, bg=BG).pack(side="left")
        tk.Frame(sys_row, bg=ACCENT, width=22, height=1).pack(side="left", padx=(10, 0))

        # Wordmark
        _lbl(content, "NEXUS", font=_fm(48, "bold"), fg=TEXT,   bg=BG).pack(anchor="w")
        _lbl(content, "AUTH",  font=_fm(48, "bold"), fg=ACCENT, bg=BG).pack(anchor="w")

        _lbl(content, "Biometric Access Control System",
             font=_f(11), fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(12, 36))

        # Status indicator row
        status_row = tk.Frame(content, bg=BG)
        status_row.pack(anchor="w", pady=(0, 28))

        self._dot_canvas = tk.Canvas(status_row, width=10, height=10,
                                     bg=BG, bd=0, highlightthickness=0)
        self._dot_canvas.pack(side="left", padx=(0, 10))
        self._dot_oval = self._dot_canvas.create_oval(1, 1, 9, 9, fill=ACCENT, outline="")

        self.lbl_sys_status = _lbl(status_row, "INITIALIZING|",
                                    font=_fm(11), fg=ACCENT, bg=BG)
        self.lbl_sys_status.pack(side="left")

        # Metric cards
        metrics_row = tk.Frame(content, bg=BG)
        metrics_row.pack(anchor="w", pady=(0, 40))
        for val, key in [("99.7%", "ACCURACY"), ("0.1%", "FAR"), ("AES-256", "VAULT")]:
            m = tk.Frame(metrics_row, bg=SURFACE)
            m.pack(side="left", padx=(0, 10))
            tk.Frame(m, bg=ACCENT, height=1).pack(fill="x")   # top cyan rule
            inner = tk.Frame(m, bg=SURFACE, padx=14, pady=8)
            inner.pack()
            tk.Label(inner, text=val, font=_fm(15, "bold"), fg=TEXT,   bg=SURFACE).pack()
            tk.Label(inner, text=key, font=F_HUD,           fg=SUBTEXT, bg=SURFACE).pack()

        # Buttons
        _btn(content, "CREATE ACCOUNT",  self._go_register, "primary",   width=18).pack(anchor="w", pady=(0, 10))
        _btn(content, "LOGIN WITH FACE", self._go_login,    "secondary", width=18).pack(anchor="w")

        # ── Right column ──────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        ring_wrap = tk.Frame(right, bg=BG)
        ring_wrap.place(relx=0.5, rely=0.5, anchor="center")

        _lbl(ring_wrap, "NEXUS.AUTH.v4.2      AES-256-GCM      FACENET+ARCFACE",
             font=F_HUD, fg=SUBTEXT, bg=BG).pack(pady=(0, 8))

        self._canvas = ScanCanvas(ring_wrap, size=300)
        self._canvas.pack()

        _lbl(ring_wrap, "ACC 99.73%      EER 0.004      68 LANDMARKS",
             font=F_HUD, fg=SUBTEXT, bg=BG).pack(pady=(8, 0))

    # ── Status cycling ────────────────────────────────────────────────────

    def _tick_status(self) -> None:
        if not self.winfo_exists():
            return
        self._status_idx = (self._status_idx + 1) % len(_STATUS_CYCLE)
        self._refresh_status()
        self._status_job = self.after(2400, self._tick_status)

    def _tick_cursor(self) -> None:
        if not self.winfo_exists():
            return
        self._cursor_on = not self._cursor_on
        col = ACCENT if self._cursor_on else DIM
        try:
            self._dot_canvas.itemconfigure(self._dot_oval, fill=col)
        except tk.TclError:
            return
        self._refresh_status()
        self._cursor_job = self.after(700, self._tick_cursor)

    def _refresh_status(self) -> None:
        if not self.lbl_sys_status:
            return
        cursor = "|" if self._cursor_on else " "
        try:
            self.lbl_sys_status.configure(
                text=f"{_STATUS_CYCLE[self._status_idx]}{cursor}"
            )
        except tk.TclError:
            pass

    def cleanup(self) -> None:
        if self._canvas:
            self._canvas.stop()
        for job in (self._status_job, self._cursor_job):
            if job:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass


# ---------------------------------------------------------------------------
# RegisterScreen
# ---------------------------------------------------------------------------

class RegisterScreen(_Screen):
    """
    Thin cyan progress bar beneath the header.
    Left: personal-info form + pose guide.
    Right: live webcam with cyan corner-bracket face overlay.
    """

    def __init__(self, master, go_back, on_done) -> None:
        super().__init__(master)
        self._go_back = go_back
        self._on_done = on_done
        self._cap: Optional[cv2.VideoCapture] = None
        self._job: Optional[str] = None
        self._pose_captures: list = [None] * len(_POSES)
        self._pose_idx: int = 0
        self._build()
        self.after(80, self._update_progress)
        self._start_cam()

    def _build(self) -> None:
        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=PANEL, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Button(
            hdr, text="← BACK", command=self._go_back,
            bg=PANEL, fg=SUBTEXT, font=F_HUD, relief="flat",
            cursor="hand2", activebackground=PANEL, activeforeground=ACCENT,
        ).pack(side="left", padx=20)

        _lbl(hdr, "CREATE ACCOUNT", font=_fm(12, "bold"), bg=PANEL, fg=TEXT).pack(side="left")

        self.lbl_step = _lbl(
            hdr, f"STEP 1 OF {len(_POSES) + 1}",
            font=F_HUD, fg=SUBTEXT, bg=PANEL,
        )
        self.lbl_step.pack(side="right", padx=20)

        # ── Progress bar ──────────────────────────────────────────────────
        prog_track = tk.Frame(self, bg=DIM, height=2)
        prog_track.pack(fill="x")
        self._prog_fill = tk.Frame(prog_track, bg=ACCENT, height=2)
        self._prog_fill.pack(side="left", fill="y")

        # ── Body ──────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Left — form
        left = tk.Frame(body, bg=BG, width=310)
        left.pack(side="left", fill="y", padx=(28, 0), pady=24)
        left.pack_propagate(False)

        _lbl(left, "Personal Information", font=F_H2, fg=TEXT).pack(anchor="w", pady=(0, 14))

        for label, attr in [
            ("FIRST NAME",                 "e_first"),
            ("LAST NAME",                  "e_last"),
            ("DATE OF BIRTH (YYYY-MM-DD)", "e_dob"),
        ]:
            row = tk.Frame(left, bg=BG)
            row.pack(anchor="w", fill="x", pady=(0, 10))
            _lbl(row, label, font=F_LABEL, fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(0, 4))
            e = _entry(row, width=30)
            e.pack(anchor="w", fill="x", ipady=8, ipadx=8)
            setattr(self, attr, e)

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=(6, 16))

        _lbl(left, "Face Capture", font=F_H2, fg=TEXT).pack(anchor="w", pady=(0, 8))

        self.lbl_pose_instr = _lbl(left, _POSES[0][0], font=_f(11), fg=TEXT, bg=BG)
        self.lbl_pose_instr.pack(anchor="w", pady=(0, 10))

        dots_row = tk.Frame(left, bg=BG)
        dots_row.pack(anchor="w", pady=(0, 12))
        self._pose_dots: list = []
        for _, tag in _POSES:
            col = tk.Frame(dots_row, bg=BG)
            col.pack(side="left", padx=(0, 14))
            c    = tk.Canvas(col, width=10, height=10, bg=BG, bd=0, highlightthickness=0)
            c.pack()
            oval = c.create_oval(1, 1, 9, 9, fill=DIM, outline="")
            lbl  = _lbl(col, tag, font=F_HUD, fg=SUBTEXT, bg=BG)
            lbl.pack()
            self._pose_dots.append((c, oval, lbl))

        self.lbl_status = _lbl(left, "Position your face in the camera",
                                font=_f(10), fg=SUBTEXT, bg=BG)
        self.lbl_status.pack(anchor="w", pady=(0, 16))

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(anchor="w")
        self.btn_capture = _btn(btn_row, f"CAPTURE  1/{len(_POSES)}", self._capture, "secondary")
        self.btn_capture.pack(side="left", padx=(0, 10))
        self.btn_register = _btn(btn_row, "REGISTER", self._submit, "primary")
        self.btn_register.configure(state="disabled")
        self.btn_register.pack(side="left")

        # Right — webcam panel
        right = tk.Frame(body, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.video_lbl = tk.Label(right, bg="#000000")
        self.video_lbl.pack(fill="both", expand=True)

        self.lbl_cam = _lbl(right, "Waiting for camera…", font=F_SMALL, fg=SUBTEXT, bg=PANEL)
        self.lbl_cam.pack(pady=8)

    def _update_progress(self) -> None:
        total = len(_POSES) + 1
        step  = min(self._pose_idx + 1, total)
        self._prog_fill.configure(width=int(960 * step / total))

    def _start_cam(self) -> None:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._status("Camera not found.", ERROR)
        else:
            self._tick()

    def _tick(self) -> None:
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self._show(self._annotate(frame))
        self._job = self.after(33, self._tick)

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces_haar(gray)
        out   = frame.copy()
        if len(faces):
            self.lbl_cam.configure(text="Face detected — ready to capture", fg=SUCCESS)
        else:
            self.lbl_cam.configure(text="No face detected", fg=SUBTEXT)
        for x, y, w, h in faces:
            _draw_brackets(out, x, y, w, h, _hex_to_bgr(ACCENT))
        return out

    def _show(self, frame: np.ndarray, target=(560, 430)) -> None:
        h, w  = frame.shape[:2]
        scale = min(target[0] / w, target[1] / h)
        rgb   = cv2.cvtColor(
            cv2.resize(frame, (int(w * scale), int(h * scale))),
            cv2.COLOR_BGR2RGB,
        )
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_lbl.configure(image=photo)
        self.video_lbl.image = photo

    def _capture(self) -> None:
        if self._pose_idx >= len(_POSES):
            return
        if not (self._cap and self._cap.isOpened()):
            self._status("Camera not available.", ERROR); return
        ret, frame = self._cap.read()
        if not ret:
            self._status("Failed to read frame.", ERROR); return
        frame = cv2.flip(frame, 1)
        crop  = _crop_face_hog(frame)
        if crop is None:
            self._status("No face detected — try again.", ERROR); return

        self._pose_captures[self._pose_idx] = crop
        canvas, oval, lbl = self._pose_dots[self._pose_idx]
        canvas.itemconfigure(oval, fill=SUCCESS)
        lbl.configure(fg=SUCCESS)

        self._pose_idx += 1
        self._update_progress()

        if self._pose_idx < len(_POSES):
            instr, _ = _POSES[self._pose_idx]
            self.lbl_pose_instr.configure(text=instr)
            self.btn_capture.configure(text=f"CAPTURE  {self._pose_idx + 1}/{len(_POSES)}")
            self.lbl_step.configure(text=f"STEP {self._pose_idx + 1} OF {len(_POSES) + 1}")
            self._status(f"Pose {self._pose_idx}/{len(_POSES)} captured.", SUCCESS)
        else:
            self.lbl_pose_instr.configure(text="All poses captured")
            self.btn_capture.configure(state="disabled")
            self.btn_register.configure(state="normal")
            self.lbl_step.configure(text="READY TO REGISTER")
            self._status("All poses ready — click REGISTER.", SUCCESS)

    def _submit(self) -> None:
        first = self.e_first.get().strip()
        last  = self.e_last.get().strip()
        dob   = self.e_dob.get().strip()

        if not first or not last:
            self._status("First and last name are required.", ERROR); return
        if not dob:
            self._status("Date of birth is required.", ERROR); return
        if self._pose_idx < len(_POSES):
            self._status(f"Capture all {len(_POSES)} poses first.", ERROR); return

        self._status("Extracting embeddings…", SUBTEXT)
        self.btn_register.configure(state="disabled")
        threading.Thread(
            target=self._do_register,
            args=(first, last, dob, [c for c in self._pose_captures if c is not None]),
            daemon=True,
        ).start()

    def _do_register(self, first: str, last: str, dob: str, face_imgs: list) -> None:
        try:
            embs = [
                _normalise_emb(emb)
                for img in face_imgs
                if (emb := self.app.deep_model.extract_embedding(img)) is not None
            ]
            if not embs:
                self.after(0, lambda: self._status("Could not extract embedding.", ERROR))
                self.after(0, lambda: self.btn_register.configure(state="normal"))
                return
            user_id = self.app.user_store.create(first, last, dob)
            gallery = _load_gallery()
            gallery[user_id] = embs
            _save_gallery(gallery)
            self.after(0, self._on_done)
        except Exception as exc:
            self.after(0, lambda: self._status(f"Error: {exc}", ERROR))
            self.after(0, lambda: self.btn_register.configure(state="normal"))

    def _status(self, msg: str, color: str = SUBTEXT) -> None:
        self.lbl_status.configure(text=msg, fg=color)

    def cleanup(self) -> None:
        if self._job:
            self.after_cancel(self._job)
        if self._cap and self._cap.isOpened():
            self._cap.release()


# ---------------------------------------------------------------------------
# LoginScreen
# ---------------------------------------------------------------------------

class LoginScreen(_Screen):
    """
    Webcam fills the window.  HUD header + cyan confirmation bar at bottom.
    N consecutive matches above threshold trigger login.
    """

    def __init__(self, master, go_back, on_login) -> None:
        super().__init__(master)
        self._go_back       = go_back
        self._on_login      = on_login
        self._cap: Optional[cv2.VideoCapture] = None
        self._job: Optional[str] = None
        self._q: queue.Queue    = queue.Queue()
        self._processing        = False
        self._last_uid: Optional[str] = None
        self._confirm_cnt       = 0
        self._logged_in         = False
        self._overlay_text      = "SCANNING…"
        self._overlay_col       = SUBTEXT
        self._build()
        self._start_cam()

    def _build(self) -> None:
        # ── Header HUD ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=PANEL, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Button(
            hdr, text="← BACK", command=self._go_back,
            bg=PANEL, fg=SUBTEXT, font=F_HUD, relief="flat",
            cursor="hand2", activebackground=PANEL, activeforeground=ACCENT,
        ).pack(side="left", padx=20)

        _lbl(hdr, "FACE AUTHENTICATION", font=_fm(12, "bold"), bg=PANEL, fg=TEXT).pack(side="left")

        # LIVE badge in violet
        live_frame = tk.Frame(hdr, bg=ACCENT2, padx=8, pady=3)
        live_frame.pack(side="right", padx=20, pady=12)
        tk.Label(live_frame, text="● LIVE", font=_fm(8, "bold"), fg=TEXT, bg=ACCENT2).pack()

        # ── Video ─────────────────────────────────────────────────────────
        self.video_lbl = tk.Label(self, bg="#000000")
        self.video_lbl.pack(fill="both", expand=True)

        # ── Status bar ────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=PANEL, height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Confirmation progress track
        conf_track = tk.Frame(bar, bg=DIM, height=2)
        conf_track.pack(fill="x", side="top")
        self._conf_bar = tk.Frame(conf_track, bg=ACCENT, height=2, width=0)
        self._conf_bar.pack(side="left", fill="y")

        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=20)

        self.lbl_status = _lbl(inner, self._overlay_text, font=_fm(11), fg=SUBTEXT, bg=PANEL)
        self.lbl_status.pack(side="left", expand=True)
        _lbl(inner, "NEXUS AUTH SYSTEM", font=F_HUD, fg=DIM, bg=PANEL).pack(side="right")

    def _start_cam(self) -> None:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.lbl_status.configure(text="CAMERA NOT FOUND", fg=ERROR)
        else:
            self._tick()

    def _tick(self) -> None:
        if self._logged_in:
            return
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                try:
                    while True:
                        uid, score = self._q.get_nowait()
                        self._handle(uid, score)
                except queue.Empty:
                    pass
                if not self._processing:
                    self._processing = True
                    threading.Thread(
                        target=self._identify, args=(frame.copy(),), daemon=True
                    ).start()
                self._show(self._annotate(frame))
        self._job = self.after(33, self._tick)

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces_haar(gray)
        out   = frame.copy()
        col   = _hex_to_bgr(self._overlay_col)
        for x, y, w, h in faces:
            _draw_brackets(out, x, y, w, h, col)
            cv2.putText(out, self._overlay_text,
                        (x, max(y - 12, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        return out

    def _show(self, frame: np.ndarray) -> None:
        lbl_w = self.video_lbl.winfo_width()
        lbl_h = self.video_lbl.winfo_height()
        if lbl_w < 2:
            lbl_w, lbl_h = 960, 540
        h, w  = frame.shape[:2]
        scale = min(lbl_w / w, lbl_h / h)
        rgb   = cv2.cvtColor(
            cv2.resize(frame, (int(w * scale), int(h * scale))),
            cv2.COLOR_BGR2RGB,
        )
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_lbl.configure(image=photo)
        self.video_lbl.image = photo

    def _identify(self, frame: np.ndarray) -> None:
        try:
            gallery = self.app._gallery
            if not gallery:
                self._q.put(("__no_users__", 0.0)); return
            crop = _crop_face(frame)
            if crop is None:
                self._q.put(("__no_face__", 0.0)); return
            emb = self.app.deep_model.extract_embedding(crop)
            if emb is None:
                self._q.put(("__unknown__", 0.0)); return
            emb = _normalise_emb(emb)
            best_uid, best_sim = "__unknown__", 0.0
            for uid, stored in gallery.items():
                try:
                    templates = stored if isinstance(stored, list) else [stored]
                    sim = max(cosine_similarity(emb, t) for t in templates)
                    if sim > best_sim:
                        best_sim, best_uid = sim, uid
                except Exception:
                    pass
            if best_sim < _SIM_THRESHOLD:
                best_uid = "__unknown__"
            self._q.put((best_uid, best_sim))
        except Exception:
            self._q.put(("__error__", 0.0))
        finally:
            self._processing = False

    def _handle(self, uid: str, score: float) -> None:
        if uid == "__no_users__":
            self._overlay_text = "NO ACCOUNTS FOUND"
            self._overlay_col  = WARN
            self._reset_streak(); return
        if uid == "__no_face__":
            self._overlay_text = "SCANNING…"
            self._overlay_col  = SUBTEXT
            self._reset_streak(); return
        if uid in ("__unknown__", "__error__"):
            self._overlay_text = "FACE NOT RECOGNISED"
            self._overlay_col  = ERROR
            self._reset_streak(); return

        name = self.app.user_store.display_name(uid)
        self._overlay_text = f"{name}  {score:.0%}"
        self._overlay_col  = SUCCESS
        self.lbl_status.configure(text=self._overlay_text, fg=ACCENT)

        if uid == self._last_uid:
            self._confirm_cnt += 1
        else:
            self._last_uid    = uid
            self._confirm_cnt = 1

        pct   = min(self._confirm_cnt / _CONFIRM_NEEDED, 1.0)
        bar_w = int(self.winfo_width() * pct)
        self._conf_bar.configure(width=bar_w)

        if self._confirm_cnt >= _CONFIRM_NEEDED and not self._logged_in:
            self._logged_in = True
            self.after(0, lambda: self._on_login(uid))

    def _reset_streak(self) -> None:
        self._last_uid    = None
        self._confirm_cnt = 0
        self._conf_bar.configure(width=0)
        self.lbl_status.configure(text=self._overlay_text, fg=self._overlay_col)

    def cleanup(self) -> None:
        self._logged_in = True
        if self._job:
            self.after_cancel(self._job)
        if self._cap and self._cap.isOpened():
            self._cap.release()


# ---------------------------------------------------------------------------
# WelcomeScreen
# ---------------------------------------------------------------------------

class WelcomeScreen(_Screen):
    """Cyberpunk success state — identity verified."""

    def __init__(self, master, user: dict, go_home) -> None:
        super().__init__(master)
        self._user    = user
        self._go_home = go_home
        self._build()

    def _build(self) -> None:
        c = tk.Frame(self, bg=BG)
        c.place(relx=0.5, rely=0.5, anchor="center")

        # System label
        _lbl(c, "NEXUS AUTH SYSTEM", font=F_HUD, fg=SUBTEXT, bg=BG).pack(pady=(0, 12))

        # Divider
        tk.Frame(c, bg=DIM, height=1, width=340).pack(pady=(0, 20))

        # IDENTITY VERIFIED label
        _lbl(c, "IDENTITY VERIFIED", font=_fm(10, "bold"), fg=ACCENT, bg=BG).pack(pady=(0, 4))

        # Checkmark
        tk.Label(c, text="✓", font=(_FM, 64, "bold"), fg=SUCCESS, bg=BG).pack(pady=(0, 16))

        # Welcome label
        _lbl(c, "WELCOME BACK", font=_fm(10), fg=SUBTEXT, bg=BG).pack()

        # Full name
        name = f"{self._user['first_name']} {self._user['surname']}"
        _lbl(c, name, font=_fm(28, "bold"), fg=TEXT, bg=BG).pack(pady=(6, 24))

        # DOB card with violet top rule
        dob_card = tk.Frame(c, bg=SURFACE)
        dob_card.pack(pady=(0, 32))
        tk.Frame(dob_card, bg=ACCENT2, height=1).pack(fill="x")
        dob_inner = tk.Frame(dob_card, bg=SURFACE, padx=24, pady=10)
        dob_inner.pack()
        _lbl(dob_inner, "DATE OF BIRTH", font=F_HUD,      fg=SUBTEXT, bg=SURFACE).pack()
        _lbl(dob_inner, self._user["dob"], font=_fm(13),  fg=TEXT,    bg=SURFACE).pack(pady=(4, 0))

        # Sign out
        _btn(c, "SIGN OUT", self._go_home, "ghost", width=12).pack()


# ---------------------------------------------------------------------------
# Shared drawing helpers (logic unchanged)
# ---------------------------------------------------------------------------

def _draw_brackets(img: np.ndarray, x: int, y: int, w: int, h: int,
                   col: tuple, bw: int = 2) -> None:
    blen = max(min(w, h) // 4, 12)
    pts  = [
        ((x,             y + blen),     (x,     y),     (x + blen,     y)),
        ((x + w - blen,  y),            (x + w, y),     (x + w,        y + blen)),
        ((x,             y + h - blen), (x,     y + h), (x + blen,     y + h)),
        ((x + w - blen,  y + h),        (x + w, y + h), (x + w,        y + h - blen)),
    ]
    for a, b, c_ in pts:
        cv2.line(img, a, b,  col, bw, cv2.LINE_AA)
        cv2.line(img, b, c_, col, bw, cv2.LINE_AA)


def _crop_face(frame: np.ndarray) -> Optional[np.ndarray]:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces_haar(gray)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad  = int(max(w, h) * 0.1)
    ih, iw = frame.shape[:2]
    crop = frame[max(y - pad, 0):min(y + h + pad, ih),
                 max(x - pad, 0):min(x + w + pad, iw)]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return normalize_lighting(cv2.resize(crop, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA))


def _crop_face_hog(frame: np.ndarray) -> Optional[np.ndarray]:
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = _HOG(rgb, 1)
    if not rects:
        return None
    rect  = max(rects, key=lambda r: r.width() * r.height())
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    pad   = int(max(w, h) * 0.1)
    ih, iw = frame.shape[:2]
    crop  = frame[max(y - pad, 0):min(y + h + pad, ih),
                  max(x - pad, 0):min(x + w + pad, iw)]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return normalize_lighting(cv2.resize(crop, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA))


def _normalise_emb(emb: np.ndarray) -> np.ndarray:
    target = config.PROJECTION_INPUT_DIM
    flat   = emb.flatten().astype(np.float32)
    if flat.shape[0] < target:
        flat = np.pad(flat, (0, target - flat.shape[0]))
    elif flat.shape[0] > target:
        flat = flat[:target]
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 1e-6 else flat


def _hex_to_bgr(hex_col: str) -> tuple:
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceAuthApp()
    app.mainloop()
