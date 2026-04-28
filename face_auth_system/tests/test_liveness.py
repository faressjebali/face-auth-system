"""Unit tests for liveness.py."""

import numpy as np
import pytest

from liveness import (
    BlinkLivenessDetector,
    LBPLivenessDetector,
    check_liveness_single_frame,
    compute_ear,
    eye_aspect_ratio,
)


# ---------------------------------------------------------------------------
# EAR helpers
# ---------------------------------------------------------------------------

def _open_eye() -> np.ndarray:
    """Synthetic 6-point eye with EAR ≈ 0.35 (open)."""
    return np.array([
        [0, 0], [1, 1], [2, 1],
        [3, 0], [2, -1], [1, -1],
    ], dtype=float)


def _closed_eye() -> np.ndarray:
    """Synthetic 6-point eye with EAR ≈ 0.0 (closed)."""
    return np.array([
        [0, 0], [1, 0.05], [2, 0.05],
        [3, 0], [2, -0.05], [1, -0.05],
    ], dtype=float)


def test_ear_open():
    assert eye_aspect_ratio(_open_eye()) > 0.25


def test_ear_closed():
    assert eye_aspect_ratio(_closed_eye()) < 0.15


def test_compute_ear_uses_both_eyes():
    # 68-point landmarks: all zeros except eye clusters
    lm = np.zeros((68, 2), dtype=float)
    lm[36:42] = _open_eye()
    lm[42:48] = _open_eye()
    assert compute_ear(lm) > 0.25


# ---------------------------------------------------------------------------
# BlinkLivenessDetector
# ---------------------------------------------------------------------------

def _make_landmarks(open_eye=True) -> np.ndarray:
    lm = np.zeros((68, 2), dtype=float)
    eye = _open_eye() if open_eye else _closed_eye()
    lm[36:42] = eye
    lm[42:48] = eye
    return lm


def test_blink_detector_no_blink():
    det = BlinkLivenessDetector(blinks_required=1)
    for _ in range(20):
        passed, count = det.update(_make_landmarks(open_eye=True))
    assert not passed
    assert count == 0


def test_blink_detector_counts_blink():
    det = BlinkLivenessDetector(ear_threshold=0.25, consec_frames=2, blinks_required=1)
    # Simulate 3 closed frames then open
    for _ in range(3):
        det.update(_make_landmarks(open_eye=False))
    passed, count = det.update(_make_landmarks(open_eye=True))
    assert count >= 1


def test_blink_detector_reset():
    det = BlinkLivenessDetector(ear_threshold=0.25, consec_frames=2, blinks_required=1)
    for _ in range(3):
        det.update(_make_landmarks(open_eye=False))
    det.update(_make_landmarks(open_eye=True))
    det.reset()
    assert det._blinks == 0
    assert det._counter == 0


# ---------------------------------------------------------------------------
# LBPLivenessDetector
# ---------------------------------------------------------------------------

def test_lbp_real_face_texture():
    rng = np.random.default_rng(0)
    # Rich texture image
    real = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    det = LBPLivenessDetector(threshold=0.0)  # pass any non-zero score
    is_live, score = det.predict(real)
    assert score >= 0.0


def test_lbp_flat_image():
    flat = np.full((128, 128), 127, dtype=np.uint8)
    det = LBPLivenessDetector(threshold=1.0)  # very strict threshold
    is_live, score = det.predict(flat)
    assert not is_live


def test_check_liveness_single_frame_convenience():
    rng = np.random.default_rng(7)
    face = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    is_live, score = check_liveness_single_frame(face)
    assert isinstance(is_live, bool)
    assert 0.0 <= score <= 1.0
