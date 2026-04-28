"""
Liveness detection module.

Two complementary checks:

1. **Blink-based** (temporal, multi-frame):  Eye Aspect Ratio (EAR) drops
   sharply during a real blink.  A printed/displayed photo cannot blink.

2. **Texture-based** (spatial, single-frame):  LBP histograms distinguish
   rich real-face textures from flat printed/screen spoofs.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import distance as dist

import config

logger = logging.getLogger(__name__)

# Dlib 68-point indices for each eye
_LEFT_EYE = list(range(36, 42))
_RIGHT_EYE = list(range(42, 48))


# ---------------------------------------------------------------------------
# EAR helpers
# ---------------------------------------------------------------------------


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    Eye Aspect Ratio for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    eye_pts : np.ndarray
        Shape (6, 2) — six (x, y) landmark coordinates for one eye.
    """
    a = dist.euclidean(eye_pts[1], eye_pts[5])
    b = dist.euclidean(eye_pts[2], eye_pts[4])
    c = dist.euclidean(eye_pts[0], eye_pts[3])
    return (a + b) / (2.0 * c + 1e-6)


def compute_ear(landmarks: np.ndarray) -> float:
    """Mean EAR across both eyes from a full 68-point landmark array."""
    left = landmarks[_LEFT_EYE]
    right = landmarks[_RIGHT_EYE]
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0


# ---------------------------------------------------------------------------
# Blink-based liveness (stateful, temporal)
# ---------------------------------------------------------------------------


class BlinkLivenessDetector:
    """
    Counts blinks across a live video stream.

    Call :meth:`update` once per frame.  Returns ``True`` once the required
    number of blinks are confirmed.

    Parameters
    ----------
    ear_threshold : float
        EAR below this triggers a blink frame.
    consec_frames : int
        Consecutive low-EAR frames required to count one blink.
    blinks_required : int
        Blinks needed to pass liveness.
    """

    def __init__(
        self,
        ear_threshold: float = config.EAR_THRESHOLD,
        consec_frames: int = config.EAR_CONSEC_FRAMES,
        blinks_required: int = config.BLINKS_REQUIRED,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.blinks_required = blinks_required
        self._counter = 0
        self._blinks = 0

    def update(self, landmarks: np.ndarray) -> Tuple[bool, int]:
        """
        Process one frame's landmarks.

        Returns
        -------
        Tuple[bool, int]
            ``(liveness_passed, blink_count)``
        """
        ear = compute_ear(landmarks)
        if ear < self.ear_threshold:
            self._counter += 1
        else:
            if self._counter >= self.consec_frames:
                self._blinks += 1
                logger.debug("Blink confirmed. Total=%d", self._blinks)
            self._counter = 0
        return self._blinks >= self.blinks_required, self._blinks

    def reset(self) -> None:
        self._counter = 0
        self._blinks = 0


# ---------------------------------------------------------------------------
# LBP-based texture liveness (single-frame)
# ---------------------------------------------------------------------------


def _lbp_image(gray: np.ndarray) -> np.ndarray:
    """Compute 8-neighbour radius-1 LBP for every pixel."""
    result = np.zeros_like(gray, dtype=np.uint8)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for shift, (dy, dx) in enumerate(neighbours):
        rolled = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        result |= ((rolled >= gray).astype(np.uint8) << shift)
    return result


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """
    3×3-grid LBP histogram (9 cells, 256 bins each → 2304-D vector).
    """
    h, w = gray.shape[:2]
    ch, cw = h // 3, w // 3
    features = []
    for r in range(3):
        for c in range(3):
            cell = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            lbp = _lbp_image(cell)
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            features.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(features)


class LBPLivenessDetector:
    """
    Single-frame texture liveness via LBP histograms.

    Without a trained classifier the detector uses a texture-variance
    heuristic: real faces have richer local texture than flat spoofs.

    Parameters
    ----------
    threshold : float
        Liveness confidence must exceed this to pass.
    """

    def __init__(self, threshold: float = config.LBP_LIVENESS_THRESHOLD) -> None:
        self.threshold = threshold
        self._clf = None

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained sklearn classifier (pickle)."""
        import pickle
        with open(model_path, "rb") as f:
            self._clf = pickle.load(f)
        logger.info("LBP liveness model loaded from '%s'.", model_path)

    def predict(self, face_gray: np.ndarray) -> Tuple[bool, float]:
        """
        Predict whether *face_gray* is a live face.

        Returns
        -------
        Tuple[bool, float]
            ``(is_live, confidence)``
        """
        hist = _lbp_histogram(face_gray)

        if self._clf is not None:
            try:
                prob = self._clf.predict_proba(hist.reshape(1, -1))[0]
                confidence = float(prob[1]) if len(prob) > 1 else float(prob[0])
                return confidence >= self.threshold, confidence
            except Exception as exc:
                logger.warning("LBP classifier error: %s — using heuristic.", exc)

        # Variance heuristic: higher variance → more real-face texture
        variance = float(np.var(hist))
        score = float(np.clip(variance / 0.005, 0.0, 1.0))
        return score >= self.threshold, score


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def check_liveness_single_frame(
    face_gray: np.ndarray,
    detector: Optional[LBPLivenessDetector] = None,
) -> Tuple[bool, float]:
    """Run texture liveness on a single grayscale face crop."""
    if detector is None:
        detector = LBPLivenessDetector()
    return detector.predict(face_gray)
