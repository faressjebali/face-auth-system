"""
Performance evaluation framework.

Computes standard biometric system metrics from a labelled score dataset:

  - ROC curve  (TPR vs FPR)
  - FAR / FRR  curves vs decision threshold
  - EER        (Equal Error Rate)
  - AUC        (Area Under the ROC Curve)
  - Accuracy, Precision, Recall at ``config.AUTH_THRESHOLD``

Plots are saved to ``config.EVALUATION_OUTPUT_DIR``.
A CSV summary is written to ``config.METRICS_CSV_PATH``.
"""

import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import config
from logger import read_logs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions (pure numpy, no sklearn dependency)
# ---------------------------------------------------------------------------


def compute_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep thresholds and compute TPR / FPR at each point.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(fpr, tpr, thresholds)``
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    fpr_arr, tpr_arr = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fpr_arr.append(fp / (fp + tn + 1e-9))
        tpr_arr.append(tp / (tp + fn + 1e-9))
    return np.array(fpr_arr), np.array(tpr_arr), thresholds


def compute_far_frr(
    labels: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FAR = FP/(FP+TN)  — False Accept Rate (impostors let through).
    FRR = FN/(FN+TP)  — False Reject Rate (genuines turned away).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(thresholds, far, frr)``
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    far_arr, frr_arr = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        far_arr.append(fp / (fp + tn + 1e-9))
        frr_arr.append(fn / (fn + tp + 1e-9))
    return thresholds, np.array(far_arr), np.array(frr_arr)


def compute_eer(
    thresholds: np.ndarray,
    far: np.ndarray,
    frr: np.ndarray,
) -> Tuple[float, float]:
    """
    Equal Error Rate: threshold where FAR ≈ FRR.

    Returns
    -------
    Tuple[float, float]
        ``(eer_rate, eer_threshold)``
    """
    idx = int(np.abs(far - frr).argmin())
    return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _roc_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Face Auth System")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        config.EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.ROC_PLOT_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC plot → %s", config.ROC_PLOT_PATH)
    except Exception as exc:
        logger.error("ROC plot failed: %s", exc)


def _far_frr_plot(
    thresholds: np.ndarray,
    far: np.ndarray,
    frr: np.ndarray,
    eer: float,
    eer_thresh: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(thresholds, far, color="crimson", lw=2, label="FAR")
        ax.plot(thresholds, frr, color="steelblue", lw=2, label="FRR")
        ax.axvline(
            eer_thresh, color="gray", ls="--", lw=1,
            label=f"EER = {eer:.4f} @ {eer_thresh:.3f}",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Error Rate")
        ax.set_title("FAR / FRR vs Threshold")
        ax.legend()
        ax.grid(alpha=0.3)
        config.EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FAR_FRR_PLOT_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("FAR/FRR plot → %s", config.FAR_FRR_PLOT_PATH)
    except Exception as exc:
        logger.error("FAR/FRR plot failed: %s", exc)


def _write_csv(metrics: dict) -> None:
    config.EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    logger.info("Metrics CSV → %s", config.METRICS_CSV_PATH)


# ---------------------------------------------------------------------------
# High-level Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """
    End-to-end evaluation: load data → compute metrics → save plots + CSV.

    Usage (from logs)::

        ev = Evaluator()
        ev.load_from_logs(days=30)
        metrics = ev.run()

    Usage (with external labels/scores)::

        ev = Evaluator(labels=np.array([1,0,1,...]), scores=np.array([0.9,0.2,...]))
        metrics = ev.run()

    Parameters
    ----------
    labels : np.ndarray, optional
        Binary ground-truth (1 = genuine, 0 = impostor).
    scores : np.ndarray, optional
        Authentication scores ∈ [0, 1].
    """

    def __init__(
        self,
        labels: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
    ) -> None:
        self.labels = labels
        self.scores = scores

    # ------------------------------------------------------------------
    def load_from_logs(self, days: int = 30) -> bool:
        """
        Reconstruct evaluation data from auth logs.

        Logs must contain ``combined_score`` and ``is_genuine`` fields.

        Returns
        -------
        bool
            ``True`` if ≥10 labelled samples were found.
        """
        today = datetime.now(tz=timezone.utc)
        logs = []
        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            logs.extend(read_logs(date_str))
        lbls, scrs = [], []
        for entry in logs:
            if "combined_score" not in entry or "is_genuine" not in entry:
                continue
            lbls.append(1 if entry["is_genuine"] else 0)
            scrs.append(float(entry["combined_score"]))

        if len(lbls) < 10:
            logger.warning(
                "Only %d labelled log entries found; need ≥10 for evaluation.", len(lbls)
            )
            return False

        self.labels = np.array(lbls, dtype=int)
        self.scores = np.array(scrs, dtype=float)
        logger.info("Loaded %d evaluation samples from logs.", len(lbls))
        return True

    # ------------------------------------------------------------------
    def run(self) -> Optional[Dict]:
        """
        Run full evaluation: compute all metrics, save plots and CSV.

        Returns
        -------
        Optional[Dict]
            Metrics dictionary, or ``None`` if data is unavailable.
        """
        if self.labels is None or self.scores is None:
            logger.error("No data. Call load_from_logs() or pass labels+scores.")
            return None
        if len(self.labels) < 2:
            logger.error("Need ≥2 samples for evaluation.")
            return None

        fpr, tpr, _ = compute_roc(self.labels, self.scores)
        thresholds, far, frr = compute_far_frr(self.labels, self.scores)
        auc = compute_auc(fpr, tpr)
        eer, eer_thresh = compute_eer(thresholds, far, frr)

        # Metrics at the configured operating threshold
        t = config.AUTH_THRESHOLD
        pred = (self.scores >= t).astype(int)
        tp = int(((pred == 1) & (self.labels == 1)).sum())
        fp = int(((pred == 1) & (self.labels == 0)).sum())
        tn = int(((pred == 0) & (self.labels == 0)).sum())
        fn = int(((pred == 0) & (self.labels == 1)).sum())
        n = len(self.labels)

        eer_idx = int(np.abs(thresholds - eer_thresh).argmin())
        metrics = {
            "n_samples": n,
            "n_genuine": int(self.labels.sum()),
            "n_impostor": int((self.labels == 0).sum()),
            "auc": round(auc, 6),
            "eer": round(eer, 6),
            "eer_threshold": round(eer_thresh, 4),
            "far_at_eer": round(float(far[eer_idx]), 6),
            "frr_at_eer": round(float(frr[eer_idx]), 6),
            "accuracy": round((tp + tn) / (n + 1e-9), 6),
            "precision": round(tp / (tp + fp + 1e-9), 6),
            "recall_tpr": round(tp / (tp + fn + 1e-9), 6),
            "operating_threshold": t,
        }

        _roc_plot(fpr, tpr, auc)
        _far_frr_plot(thresholds, far, frr, eer, eer_thresh)
        _write_csv(metrics)

        logger.info("Evaluation done. AUC=%.4f  EER=%.4f @ t=%.3f", auc, eer, eer_thresh)
        return metrics
