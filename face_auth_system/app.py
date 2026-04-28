"""
Face Auth System — Desktop Application

Four screens:
  HomeScreen     → Split layout: wordmark + buttons | animated biometric scan
  RegisterScreen → Header progress + personal form | live webcam
  LoginScreen    → Full-window webcam with HUD overlay and confirmation bar
  WelcomeScreen  → Post-login success state

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
import numpy as np
from PIL import Image, ImageTk

import config
from deep_model import DeepFaceModel, cosine_similarity
from enrollment import _load_gallery, _save_gallery
from lighting import normalize_lighting
from user_store import UserStore

# ---------------------------------------------------------------------------
# Design tokens — deep charcoal + amber
# ---------------------------------------------------------------------------
BG        = "#0c0b08"   # near-black, barely-warm tint
PANEL     = "#17150f"   # raised surface
SURFACE   = "#211e15"   # elevated (inputs, wells)
ACCENT    = "#e8a92c"   # amber — primary action
ACCENT_D  = "#c48820"   # amber dark — active / pressed
TEXT      = "#f0ead8"   # warm off-white
SUBTEXT   = "#7e7566"   # warm mid-gray
DIM       = "#4a4438"   # dimmed / disabled
SUCCESS   = "#4dc87a"   # green
ERROR     = "#e05050"   # red
WARN      = "#e87828"   # orange (distinct from amber)
BORDER    = "#2e2a1e"   # subtle border
BORDER_F  = "#524a34"   # focused border (amber family)

_FF  = "Ubuntu"         # primary face
_FFM = "Ubuntu Mono"

def _f(size: int, weight: str = "normal") -> tuple:
    return (_FF, size, weight)


F_DISPLAY = _f(52, "bold")
F_TITLE   = _f(22, "bold")
F_H2      = _f(14, "bold")
F_BODY    = _f(11)
F_SMALL   = _f(10)
F_BTN     = _f(11, "bold")
F_LABEL   = _f(8,  "bold")

# Login confirmation
_CONFIRM_NEEDED = 2
_SIM_THRESHOLD  = 1.0 - config.DEEP_DISTANCE_THRESHOLD

# Multi-pose enrollment
_POSES = [
    ("Look straight at the camera", "FRONTAL"),
    ("Turn slightly left",          "LEFT"),
    ("Turn slightly right",         "RIGHT"),
]

# Shared Haar cascade
_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def _detect_faces_haar(gray: np.ndarray):
    return _HAAR.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def _btn(parent, text, command, style: str = "primary", width: int = 0, **kw):
    """
    style: "primary" | "secondary" | "ghost"
    """
    palettes = {
        "primary":   dict(bg=ACCENT,   fg="#0c0b08",  abg=ACCENT_D, afg="#0c0b08"),
        "secondary": dict(bg=SURFACE,  fg=TEXT,       abg=BORDER_F, afg=TEXT),
        "ghost":     dict(bg=BG,       fg=SUBTEXT,    abg=SURFACE,  afg=TEXT),
    }
    p = palettes.get(style, palettes["primary"])
    b = tk.Button(
        parent, text=text, command=command,
        bg=p["bg"], fg=p["fg"],
        activebackground=p["abg"], activeforeground=p["afg"],
        font=F_BTN, relief="flat", cursor="hand2",
        padx=22, pady=11,
        **kw,
    )
    if width:
        b.configure(width=width)
    return b


def _lbl(parent, text, font=F_BODY, fg=TEXT, bg=BG, **kw):
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
# Biometric scan canvas — animated arc + face guide (HomeScreen decoration)
# ---------------------------------------------------------------------------

class ScanCanvas(tk.Canvas):
    """Rotating sweep arc over a static biometric face-guide graphic."""

    def __init__(self, parent, size: int = 300, **kw):
        super().__init__(
            parent, width=size, height=size,
            bg=BG, bd=0, highlightthickness=0, **kw,
        )
        self._size  = size
        self._cx    = size // 2
        self._cy    = size // 2
        self._angle = 0
        self._job   = None
        self._draw_static()
        self._arc = self.create_arc(
            self._cx - self._r_outer, self._cy - self._r_outer,
            self._cx + self._r_outer, self._cy + self._r_outer,
            start=0, extent=55,
            style="arc", outline=ACCENT, width=2,
        )
        self._animate()

    def _draw_static(self) -> None:
        cx, cy = self._cx, self._cy
        s = self._size
        r_outer = s // 2 - 14
        r_inner = r_outer - 20
        self._r_outer = r_outer

        # Rings
        self.create_oval(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                         outline=BORDER, width=1)
        self.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                         outline=BORDER, width=1)

        # Face-guide bracket corners
        half = r_inner * 0.64
        blen = 20
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bx = cx + sx * half
            by = cy + sy * half
            # Horizontal arm
            self.create_line(bx, by, bx - sx * blen, by,
                             fill=ACCENT, width=2)
            # Vertical arm
            self.create_line(bx, by, bx, by - sy * blen,
                             fill=ACCENT, width=2)

        # Landmark dots — approximate facial keypoints
        for dx, dy in [
            (0, -0.34), (-0.38, -0.02), (0.38, -0.02),
            (-0.22, -0.16), (0.22, -0.16),
            (0, 0.46),
        ]:
            px = cx + dx * half * 1.4
            py = cy + dy * half * 1.4
            self.create_oval(px - 2, py - 2, px + 2, py + 2,
                             fill=DIM, outline="")

        # Tick marks around outer ring
        for i in range(36):
            a = math.radians(i * 10)
            tick_in = r_outer - (5 if i % 9 == 0 else 3)
            x1 = cx + r_outer * math.cos(a)
            y1 = cy + r_outer * math.sin(a)
            x2 = cx + tick_in  * math.cos(a)
            y2 = cy + tick_in  * math.sin(a)
            self.create_line(x1, y1, x2, y2, fill=BORDER, width=1)

    def _animate(self) -> None:
        self._angle = (self._angle + 4) % 360
        self.itemconfigure(self._arc, start=self._angle)
        self._job = self.after(33, self._animate)

    def stop(self) -> None:
        if self._job:
            self.after_cancel(self._job)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

class FaceAuthApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Auth")
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

    # -- Navigation --------------------------------------------------------
    def _go_home(self) -> None:
        self._set(HomeScreen(self, go_register=self._go_register, go_login=self._go_login))

    def _go_register(self) -> None:
        self._set(RegisterScreen(self, go_back=self._go_home, on_done=self._after_register))

    def _go_login(self) -> None:
        self._gallery = _load_gallery()
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
    Left: wordmark + tagline + two action buttons.
    Right: animated biometric scan visualization.
    """

    def __init__(self, master, go_register, go_login) -> None:
        super().__init__(master)
        self._go_register = go_register
        self._go_login    = go_login
        self._canvas: Optional[ScanCanvas] = None
        self._build()

    def _build(self) -> None:
        # ── Left column ───────────────────────────────────────────────────
        left = tk.Frame(self, bg=BG, width=460)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        content = tk.Frame(left, bg=BG)
        content.place(relx=0.13, rely=0.5, anchor="w")

        # Eyebrow label
        _lbl(content, "BIOMETRIC IDENTITY VERIFICATION",
             font=_f(8, "bold"), fg=ACCENT, bg=BG).pack(anchor="w")

        # Thin amber rule
        tk.Frame(content, bg=ACCENT, height=2, width=32).pack(anchor="w", pady=(8, 16))

        # Display heading — two lines for visual weight
        _lbl(content, "FACE", font=_f(52, "bold"), fg=TEXT,   bg=BG).pack(anchor="w", pady=0)
        _lbl(content, "AUTH", font=_f(52, "bold"), fg=ACCENT, bg=BG).pack(anchor="w")

        # Tagline
        _lbl(content,
             "Secure, passwordless authentication\nthrough facial recognition.",
             font=_f(11), fg=SUBTEXT, bg=BG, justify="left",
             ).pack(anchor="w", pady=(20, 40))

        # Buttons — clear hierarchy: primary + secondary
        _btn(content, "Create Account",  self._go_register, "primary",   width=18).pack(anchor="w", pady=(0, 10))
        _btn(content, "Login with Face", self._go_login,    "secondary", width=18).pack(anchor="w")

        # ── Right column ──────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        scan_wrap = tk.Frame(right, bg=BG)
        scan_wrap.place(relx=0.5, rely=0.5, anchor="center")

        self._canvas = ScanCanvas(scan_wrap, size=290)
        self._canvas.pack()

        _lbl(scan_wrap, "SYSTEM READY", font=_f(8), fg=DIM, bg=BG).pack(pady=(14, 0))

    def cleanup(self) -> None:
        if self._canvas:
            self._canvas.stop()


# ---------------------------------------------------------------------------
# RegisterScreen
# ---------------------------------------------------------------------------

class RegisterScreen(_Screen):
    """
    Thin amber progress bar beneath the header.
    Left: personal-info form + pose guide.
    Right: live webcam with corner-bracket face overlay.
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
            hdr, text="← Back", command=self._go_back,
            bg=PANEL, fg=SUBTEXT, font=F_SMALL, relief="flat",
            cursor="hand2", activebackground=PANEL, activeforeground=TEXT,
        ).pack(side="left", padx=20, pady=0)

        _lbl(hdr, "Create Account", font=F_H2, bg=PANEL, fg=TEXT).pack(side="left")

        self.lbl_step = _lbl(
            hdr, f"Step 1 of {len(_POSES) + 1}",
            font=_f(9), fg=SUBTEXT, bg=PANEL,
        )
        self.lbl_step.pack(side="right", padx=20)

        # ── Progress track ────────────────────────────────────────────────
        prog_track = tk.Frame(self, bg=BORDER, height=2)
        prog_track.pack(fill="x")
        self._prog_fill = tk.Frame(prog_track, bg=ACCENT, height=2)
        self._prog_fill.pack(side="left", fill="y")

        # ── Body ──────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Left — form + controls
        left = tk.Frame(body, bg=BG, width=310)
        left.pack(side="left", fill="y", padx=(28, 0), pady=24)
        left.pack_propagate(False)

        _lbl(left, "Personal Information", font=F_H2, fg=TEXT).pack(anchor="w", pady=(0, 14))

        for label, attr in [
            ("FIRST NAME",   "e_first"),
            ("LAST NAME",    "e_last"),
            ("DATE OF BIRTH  (YYYY-MM-DD)", "e_dob"),
        ]:
            row = tk.Frame(left, bg=BG)
            row.pack(anchor="w", fill="x", pady=(0, 10))
            _lbl(row, label, font=F_LABEL, fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(0, 4))
            e = _entry(row, width=30)
            e.pack(anchor="w", fill="x", ipady=8, ipadx=8)
            setattr(self, attr, e)

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=(6, 16))

        # Pose guide
        _lbl(left, "Face Capture", font=F_H2, fg=TEXT).pack(anchor="w", pady=(0, 8))

        self.lbl_pose_instr = _lbl(left, _POSES[0][0], font=_f(11), fg=TEXT, bg=BG)
        self.lbl_pose_instr.pack(anchor="w", pady=(0, 10))

        # Pose dots
        dots_row = tk.Frame(left, bg=BG)
        dots_row.pack(anchor="w", pady=(0, 12))
        self._pose_dots: list = []
        for _, tag in _POSES:
            col = tk.Frame(dots_row, bg=BG)
            col.pack(side="left", padx=(0, 14))
            c = tk.Canvas(col, width=10, height=10, bg=BG, bd=0, highlightthickness=0)
            c.pack()
            oval = c.create_oval(1, 1, 9, 9, fill=BORDER, outline="")
            lbl = _lbl(col, tag, font=_f(8), fg=DIM, bg=BG)
            lbl.pack()
            self._pose_dots.append((c, oval, lbl))

        # Status
        self.lbl_status = _lbl(left, "Position your face in the camera",
                                font=_f(10), fg=SUBTEXT, bg=BG)
        self.lbl_status.pack(anchor="w", pady=(0, 16))

        # Action buttons
        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(anchor="w")
        self.btn_capture = _btn(btn_row, f"Capture  1 / {len(_POSES)}", self._capture, "secondary")
        self.btn_capture.pack(side="left", padx=(0, 10))
        self.btn_register = _btn(btn_row, "Register", self._submit, "primary")
        self.btn_register.configure(state="disabled")
        self.btn_register.pack(side="left")

        # Right — webcam panel
        right = tk.Frame(body, bg=PANEL)
        right.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.video_lbl = tk.Label(right, bg="#000000")
        self.video_lbl.pack(fill="both", expand=True)

        self.lbl_cam = _lbl(right, "Waiting for camera…",
                             font=F_SMALL, fg=SUBTEXT, bg=PANEL)
        self.lbl_cam.pack(pady=8)

    # -- Progress ----------------------------------------------------------
    def _update_progress(self) -> None:
        total = len(_POSES) + 1
        step  = min(self._pose_idx + 1, total)
        bar_w = int(960 * step / total)
        self._prog_fill.configure(width=bar_w)

    # -- Camera ------------------------------------------------------------
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

    # -- Capture -----------------------------------------------------------
    def _capture(self) -> None:
        if self._pose_idx >= len(_POSES):
            return
        if not (self._cap and self._cap.isOpened()):
            self._status("Camera not available.", ERROR); return
        ret, frame = self._cap.read()
        if not ret:
            self._status("Failed to read frame.", ERROR); return
        frame = cv2.flip(frame, 1)
        crop  = _crop_face(frame)
        if crop is None:
            self._status("No face detected — try again.", ERROR); return

        self._pose_captures[self._pose_idx] = crop

        # Mark dot as captured
        canvas, oval, lbl = self._pose_dots[self._pose_idx]
        canvas.itemconfigure(oval, fill=SUCCESS)
        lbl.configure(fg=SUCCESS)

        self._pose_idx += 1
        self._update_progress()

        if self._pose_idx < len(_POSES):
            instr, _ = _POSES[self._pose_idx]
            self.lbl_pose_instr.configure(text=instr)
            self.btn_capture.configure(text=f"Capture  {self._pose_idx + 1} / {len(_POSES)}")
            self.lbl_step.configure(text=f"Step {self._pose_idx + 1} of {len(_POSES) + 1}")
            self._status(f"Pose {self._pose_idx} of {len(_POSES)} captured.", SUCCESS)
        else:
            self.lbl_pose_instr.configure(text="All poses captured")
            self.btn_capture.configure(state="disabled")
            self.btn_register.configure(state="normal")
            self.lbl_step.configure(text="Ready to register")
            self._status("All poses ready — click Register.", SUCCESS)

    # -- Register ----------------------------------------------------------
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
    Webcam fills the window.  HUD header + amber confirmation bar at bottom.
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
        self._overlay_text      = "Scanning…"
        self._overlay_col       = SUBTEXT
        self._build()
        self._start_cam()

    def _build(self) -> None:
        # ── Header HUD ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=PANEL, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Button(
            hdr, text="← Back", command=self._go_back,
            bg=PANEL, fg=SUBTEXT, font=F_SMALL, relief="flat",
            cursor="hand2", activebackground=PANEL, activeforeground=TEXT,
        ).pack(side="left", padx=20)

        _lbl(hdr, "Face Authentication", font=F_H2, bg=PANEL, fg=TEXT).pack(side="left")

        # "LIVE" badge
        live = tk.Frame(hdr, bg=ERROR)
        live.pack(side="right", padx=20, pady=12)
        tk.Label(live, text="  LIVE  ", font=_f(8, "bold"),
                 fg=TEXT, bg=ERROR, padx=0, pady=2).pack()

        # ── Video ─────────────────────────────────────────────────────────
        self.video_lbl = tk.Label(self, bg="#000000")
        self.video_lbl.pack(fill="both", expand=True)

        # ── Status bar ────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=PANEL, height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Confirmation progress track — top of bar
        conf_track = tk.Frame(bar, bg=BORDER, height=2)
        conf_track.pack(fill="x", side="top")
        self._conf_bar = tk.Frame(conf_track, bg=ACCENT, height=2, width=0)
        self._conf_bar.pack(side="left", fill="y")

        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=20)

        self.lbl_status = _lbl(inner, self._overlay_text, font=_f(11), fg=SUBTEXT, bg=PANEL)
        self.lbl_status.pack(side="left", pady=0, expand=True)
        _lbl(inner, "FACE AUTH", font=_f(8), fg=DIM, bg=PANEL).pack(side="right")

    # -- Camera ------------------------------------------------------------
    def _start_cam(self) -> None:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.lbl_status.configure(text="Camera not found.", fg=ERROR)
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
                    threading.Thread(target=self._identify,
                                     args=(frame.copy(),), daemon=True).start()
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1,
                        cv2.LINE_AA)
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

    # -- Identification ----------------------------------------------------
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
            self._overlay_text = "No accounts found"
            self._overlay_col  = WARN
            self._reset_streak(); return
        if uid == "__no_face__":
            self._overlay_text = "Scanning…"
            self._overlay_col  = SUBTEXT
            self._reset_streak(); return
        if uid in ("__unknown__", "__error__"):
            self._overlay_text = "Face not recognised"
            self._overlay_col  = ERROR
            self._reset_streak(); return

        name = self.app.user_store.display_name(uid)
        self._overlay_text = f"{name}  {score:.0%}"
        self._overlay_col  = SUCCESS
        self.lbl_status.configure(text=self._overlay_text, fg=SUCCESS)

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
    """Clean success state — minimal, typographically confident."""

    def __init__(self, master, user: dict, go_home) -> None:
        super().__init__(master)
        self._user    = user
        self._go_home = go_home
        self._build()

    def _build(self) -> None:
        c = tk.Frame(self, bg=BG)
        c.place(relx=0.5, rely=0.5, anchor="center")

        # Check mark
        tk.Label(c, text="✓", font=(_FF, 76, "bold"),
                 fg=SUCCESS, bg=BG).pack(pady=(0, 4))

        # Status label
        _lbl(c, "IDENTITY VERIFIED", font=_f(8, "bold"), fg=SUCCESS, bg=BG).pack(pady=(0, 12))

        # Horizontal rule
        tk.Frame(c, bg=BORDER, height=1, width=300).pack(pady=(0, 20))

        # Name
        _lbl(c, "Welcome back,", font=_f(12), fg=SUBTEXT, bg=BG).pack()
        name = f"{self._user['first_name']} {self._user['surname']}"
        _lbl(c, name, font=_f(30, "bold"), fg=TEXT, bg=BG).pack(pady=(4, 24))

        # DOB detail
        detail = tk.Frame(c, bg=BG)
        detail.pack(pady=(0, 36))
        _lbl(detail, "DATE OF BIRTH", font=_f(8, "bold"), fg=DIM, bg=BG).pack()
        _lbl(detail, self._user["dob"], font=_f(13), fg=SUBTEXT, bg=BG).pack(pady=(4, 0))

        # Sign-out
        _btn(c, "Sign Out", self._go_home, "ghost", width=12).pack()


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _draw_brackets(img: np.ndarray, x: int, y: int, w: int, h: int,
                   col: tuple, bw: int = 2) -> None:
    """Draw corner-bracket face overlay (premium look vs full rectangle)."""
    blen = max(min(w, h) // 4, 12)
    pts = [
        # TL
        ((x,         y + blen), (x, y), (x + blen, y)),
        # TR
        ((x + w - blen, y),     (x + w, y), (x + w, y + blen)),
        # BL
        ((x,         y + h - blen), (x, y + h), (x + blen, y + h)),
        # BR
        ((x + w - blen, y + h), (x + w, y + h), (x + w, y + h - blen)),
    ]
    for a, b, c_ in pts:
        cv2.line(img, a, b, col, bw, cv2.LINE_AA)
        cv2.line(img, b, c_, col, bw, cv2.LINE_AA)


def _crop_face(frame: np.ndarray) -> Optional[np.ndarray]:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces_haar(gray)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(max(w, h) * 0.1)
    ih, iw = frame.shape[:2]
    crop = frame[max(y - pad, 0):min(y + h + pad, ih),
                 max(x - pad, 0):min(x + w + pad, iw)]
    crop = cv2.resize(crop, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return normalize_lighting(crop)


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
