"""Unit tests for deepfake_detector.py."""

import numpy as np
import pytest

from deepfake_detector import (
    DeepfakeDetector,
    _checkerboard_score,
    _high_freq_ratio,
    _log_spectrum,
    _spectral_flatness,
)


@pytest.fixture
def natural_image():
    """Low-frequency dominant image (cosine-based, real-world-like)."""
    x = np.linspace(0, np.pi, 128)
    y = np.linspace(0, np.pi, 128)
    xx, yy = np.meshgrid(x, y)
    img = (127.5 + 127.5 * np.cos(xx) * np.cos(yy)).astype(np.uint8)
    return img


@pytest.fixture
def checkerboard_image():
    """Pure Nyquist-frequency checkerboard — maximally synthetic-looking."""
    board = np.zeros((128, 128), dtype=np.uint8)
    board[::2, ::2] = 255
    board[1::2, 1::2] = 255
    return board


def test_log_spectrum_shape(natural_image):
    spec = _log_spectrum(natural_image)
    assert spec.shape == natural_image.shape


def test_high_freq_ratio_range(natural_image, checkerboard_image):
    nat_hf = _high_freq_ratio(_log_spectrum(natural_image))
    cb_hf = _high_freq_ratio(_log_spectrum(checkerboard_image))
    assert 0.0 <= nat_hf <= 1.0
    assert 0.0 <= cb_hf <= 1.0
    # Checkerboard should have more high-frequency energy
    assert cb_hf > nat_hf


def test_checkerboard_score_higher_for_synthetic(natural_image, checkerboard_image):
    nat_score = _checkerboard_score(_log_spectrum(natural_image))
    cb_score = _checkerboard_score(_log_spectrum(checkerboard_image))
    assert cb_score >= nat_score


def test_detector_returns_tuple(natural_image):
    det = DeepfakeDetector()
    # Feed as BGR (3-channel)
    bgr = np.stack([natural_image, natural_image, natural_image], axis=-1)
    is_fake, score = det.predict(bgr)
    assert isinstance(is_fake, bool)
    assert 0.0 <= score <= 1.0


def test_is_real_inverse(natural_image):
    det = DeepfakeDetector()
    is_fake, fake_score = det.predict(natural_image)
    is_real, real_score = det.is_real(natural_image)
    assert is_real == (not is_fake)
    assert abs(fake_score + real_score - 1.0) < 1e-6


def test_broken_input_does_not_raise():
    det = DeepfakeDetector()
    # Empty image — should return (False, 0.0) gracefully
    bad = np.array([], dtype=np.uint8)
    is_fake, score = det.predict(bad)
    assert not is_fake
    assert score == 0.0
