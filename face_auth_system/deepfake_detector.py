"""
Deepfake / synthetic face detection via FFT spectral analysis.

GAN-generated faces carry characteristic high-frequency artifacts from
transposed-convolution upsampling (checkerboard patterns).  Three spectral
features are combined into a single fakeness score:

  1. High-frequency energy ratio  — GAN artifacts boost HF energy.
  2. Checkerboard score           — energy near Nyquist corners.
  3. Radial spectral flatness     — real faces have a natural 1/f roll-off;
                                    GANs often produce a flatter spectrum.

Reference: "Thinking in Frequency: Face Forgery Detection by Mining
Frequency-aware Clues" (Li et al., 2021).
"""

import logging
from typing import Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectral feature helpers
# ---------------------------------------------------------------------------


def _log_spectrum(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray.astype(np.float32))
    return np.log1p(np.abs(np.fft.fftshift(f)))


def _high_freq_ratio(spectrum: np.ndarray, lf_radius: float = 0.3) -> float:
    """Fraction of energy outside the central low-frequency disc."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * lf_radius)
    y, x = np.ogrid[:h, :w]
    lf_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    lf_energy = float(spectrum[lf_mask].sum())
    total = float(spectrum.sum()) + 1e-6
    return 1.0 - lf_energy / total


def _checkerboard_score(spectrum: np.ndarray) -> float:
    """Energy in Nyquist corner regions relative to mean (GAN upsampling leak)."""
    h, w = spectrum.shape
    sz = max(2, min(h, w) // 16)
    corners = [
        spectrum[:sz, :sz],
        spectrum[:sz, -sz:],
        spectrum[-sz:, :sz],
        spectrum[-sz:, -sz:],
    ]
    corner_mean = sum(c.mean() for c in corners) / 4.0
    return float(np.clip(corner_mean / (spectrum.mean() + 1e-6), 0.0, 1.0))


def _spectral_flatness(spectrum: np.ndarray) -> float:
    """
    Normalised score: high flatness → more GAN-like.

    Real faces follow a ~1/f roll-off (std/mean ≈ 0.5–1.5).
    GANs produce a flatter radial spectrum (std/mean < 0.3).
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    radial = np.array(
        [spectrum[r == i].mean() if (r == i).any() else 0.0 for i in range(max_r)],
        dtype=np.float32,
    )
    flatness = float(np.std(radial) / (np.mean(radial) + 1e-6))
    # Low flatness → likely GAN → high score
    return float(np.clip(1.0 - flatness / 1.5, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class DeepfakeDetector:
    """
    FFT-based synthetic face / deepfake detector.

    Parameters
    ----------
    threshold : float
        Combined fakeness score above which the image is flagged as synthetic.
    """

    def __init__(self, threshold: float = config.FFT_ARTIFACT_THRESHOLD) -> None:
        self.threshold = threshold

    def predict(self, face_img: np.ndarray) -> Tuple[bool, float]:
        """
        Score a face image for deepfake characteristics.

        Parameters
        ----------
        face_img : np.ndarray
            BGR or grayscale uint8 face crop.

        Returns
        -------
        Tuple[bool, float]
            ``(is_fake, score)`` — *score* ∈ [0, 1], higher = more synthetic.
        """
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if face_img.ndim == 3 else face_img.copy()
            gray = cv2.resize(gray, (128, 128))
            spectrum = _log_spectrum(gray)

            hf = _high_freq_ratio(spectrum)
            cb = _checkerboard_score(spectrum)
            fl = _spectral_flatness(spectrum)

            score = float(np.clip(0.4 * hf + 0.4 * cb + 0.2 * fl, 0.0, 1.0))
            is_fake = score >= self.threshold

            logger.debug(
                "Deepfake score=%.4f (hf=%.3f cb=%.3f flat=%.3f) is_fake=%s",
                score, hf, cb, fl, is_fake,
            )
            return is_fake, score

        except Exception as exc:
            logger.error("Deepfake detection error: %s", exc)
            return False, 0.0

    def is_real(self, face_img: np.ndarray) -> Tuple[bool, float]:
        """Inverse of :meth:`predict`: returns ``(is_real, authenticity_score)``."""
        is_fake, score = self.predict(face_img)
        return not is_fake, 1.0 - score
