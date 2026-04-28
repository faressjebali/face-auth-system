"""Unit tests for evaluation.py."""

import numpy as np
import pytest

from evaluation import (
    Evaluator,
    compute_auc,
    compute_eer,
    compute_far_frr,
    compute_roc,
)


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_data():
    """Genuines score 1.0, impostors score 0.0 → perfect separation."""
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([1.0, 0.9, 0.85, 0.1, 0.05, 0.0])
    return labels, scores


@pytest.fixture
def random_data():
    rng = np.random.default_rng(42)
    n = 200
    labels = rng.integers(0, 2, n)
    scores = rng.random(n).astype(float)
    return labels, scores


# ---------------------------------------------------------------------------
# compute_roc
# ---------------------------------------------------------------------------

def test_roc_shapes(random_data):
    fpr, tpr, thresholds = compute_roc(*random_data, n_thresholds=100)
    assert fpr.shape == tpr.shape == thresholds.shape == (100,)


def test_roc_perfect(perfect_data):
    fpr, tpr, _ = compute_roc(*perfect_data)
    # At some threshold the TPR should be ~1 with FPR ~0
    assert tpr.max() > 0.9
    assert fpr.min() < 0.1


def test_roc_values_in_range(random_data):
    fpr, tpr, _ = compute_roc(*random_data)
    assert np.all((fpr >= 0) & (fpr <= 1))
    assert np.all((tpr >= 0) & (tpr <= 1))


# ---------------------------------------------------------------------------
# compute_far_frr
# ---------------------------------------------------------------------------

def test_far_frr_shapes(random_data):
    thresholds, far, frr = compute_far_frr(*random_data, n_thresholds=100)
    assert thresholds.shape == far.shape == frr.shape == (100,)


def test_far_frr_ranges(random_data):
    _, far, frr = compute_far_frr(*random_data)
    assert np.all((far >= 0) & (far <= 1))
    assert np.all((frr >= 0) & (frr <= 1))


# ---------------------------------------------------------------------------
# compute_eer
# ---------------------------------------------------------------------------

def test_eer_perfect(perfect_data):
    thresholds, far, frr = compute_far_frr(*perfect_data)
    eer, eer_t = compute_eer(thresholds, far, frr)
    assert eer < 0.05   # near 0 for perfect data


def test_eer_random(random_data):
    thresholds, far, frr = compute_far_frr(*random_data)
    eer, eer_t = compute_eer(thresholds, far, frr)
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= eer_t <= 1.0


# ---------------------------------------------------------------------------
# compute_auc
# ---------------------------------------------------------------------------

def test_auc_perfect(perfect_data):
    fpr, tpr, _ = compute_roc(*perfect_data)
    auc = compute_auc(fpr, tpr)
    assert auc > 0.95


def test_auc_range(random_data):
    fpr, tpr, _ = compute_roc(*random_data)
    auc = compute_auc(fpr, tpr)
    assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def test_evaluator_run_produces_metrics(perfect_data, tmp_path, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg, "EVALUATION_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "ROC_PLOT_PATH", tmp_path / "roc.png")
    monkeypatch.setattr(cfg, "FAR_FRR_PLOT_PATH", tmp_path / "far_frr.png")
    monkeypatch.setattr(cfg, "METRICS_CSV_PATH", tmp_path / "metrics.csv")

    labels, scores = perfect_data
    ev = Evaluator(labels=labels, scores=scores)
    metrics = ev.run()
    assert metrics is not None
    assert "auc" in metrics
    assert "eer" in metrics
    assert metrics["auc"] > 0.9


def test_evaluator_no_data_returns_none():
    ev = Evaluator()
    assert ev.run() is None


def test_evaluator_metrics_csv_written(perfect_data, tmp_path, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg, "EVALUATION_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "ROC_PLOT_PATH", tmp_path / "roc.png")
    monkeypatch.setattr(cfg, "FAR_FRR_PLOT_PATH", tmp_path / "far_frr.png")
    monkeypatch.setattr(cfg, "METRICS_CSV_PATH", tmp_path / "metrics.csv")

    labels, scores = perfect_data
    ev = Evaluator(labels=labels, scores=scores)
    ev.run()
    assert (tmp_path / "metrics.csv").exists()
