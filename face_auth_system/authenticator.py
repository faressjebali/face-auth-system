"""
Score fusion and authentication decision module.

Combines classical (Eigenfaces / LBPH) and deep-learning (FaceNet / ArcFace)
confidence scores using configurable weighted fusion, then applies an adaptive
threshold to produce a binary GRANTED / DENIED decision.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AuthResult:
    """
    Container for a single authentication decision.

    Attributes
    ----------
    decision : str
        ``"GRANTED"`` or ``"DENIED"``.
    combined_score : float
        Weighted fusion score in [0, 1].
    classical_score : float
        Score from the classical pipeline in [0, 1].
    deep_score : float
        Score from the deep-learning pipeline in [0, 1].
    matched_identity : str
        User identifier returned by the recognition pipeline.
    detail : str
        Human-readable explanation of the decision.
    """

    decision: str
    combined_score: float
    classical_score: float
    deep_score: float
    matched_identity: str = "unknown"
    detail: str = ""


# ---------------------------------------------------------------------------
# Authenticator
# ---------------------------------------------------------------------------


class Authenticator:
    """
    Weighted score fusion authenticator.

    Parameters
    ----------
    classical_weight : float
        Weight assigned to the classical pipeline score (default 0.4).
    deep_weight : float
        Weight assigned to the deep pipeline score (default 0.6).
    threshold : float
        Fused score must meet or exceed this value for ``GRANTED``
        (default 0.6).

    Notes
    -----
    Weights are automatically normalised so that
    ``classical_weight + deep_weight == 1.0``.
    """

    def __init__(
        self,
        classical_weight: float = config.CLASSICAL_WEIGHT,
        deep_weight: float = config.DEEP_WEIGHT,
        threshold: float = config.AUTH_THRESHOLD,
    ) -> None:
        total = classical_weight + deep_weight
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.classical_weight = classical_weight / total
        self.deep_weight = deep_weight / total
        self.threshold = float(np.clip(threshold, 0.0, 1.0))
        logger.info(
            "Authenticator: classical_w=%.2f deep_w=%.2f threshold=%.2f",
            self.classical_weight,
            self.deep_weight,
            self.threshold,
        )

    # ------------------------------------------------------------------
    def fuse_scores(
        self,
        classical_score: float,
        deep_score: float,
    ) -> float:
        """
        Compute the weighted fusion score.

        Parameters
        ----------
        classical_score : float
            Confidence from the classical model in [0, 1].
        deep_score : float
            Similarity score from the deep model in [0, 1].

        Returns
        -------
        float
            Combined score in [0, 1].
        """
        c = float(np.clip(classical_score, 0.0, 1.0))
        d = float(np.clip(deep_score, 0.0, 1.0))
        fused = self.classical_weight * c + self.deep_weight * d
        return float(np.clip(fused, 0.0, 1.0))

    # ------------------------------------------------------------------
    def authenticate(
        self,
        classical_score: float,
        deep_score: float,
        deep_identity: Optional[str] = None,
    ) -> AuthResult:
        """
        Decide whether to grant access based on fused recognition scores.

        Parameters
        ----------
        classical_score : float
            Confidence from classical pipeline in [0, 1].
        deep_score : float
            Similarity from deep-learning pipeline in [0, 1].
        deep_identity : str, optional
            Best-match identity returned by the deep model gallery lookup.

        Returns
        -------
        AuthResult
            Full decision record.
        """
        combined = self.fuse_scores(classical_score, deep_score)
        granted = combined >= self.threshold

        decision = "GRANTED" if granted else "DENIED"
        matched = deep_identity or "unknown"

        detail = (
            f"fused={combined:.3f} threshold={self.threshold:.3f} "
            f"classical={classical_score:.3f} deep={deep_score:.3f}"
        )
        logger.info("Auth decision: %s — %s", decision, detail)

        return AuthResult(
            decision=decision,
            combined_score=round(combined, 4),
            classical_score=round(float(classical_score), 4),
            deep_score=round(float(deep_score), 4),
            matched_identity=matched,
            detail=detail,
        )

    # ------------------------------------------------------------------
    def update_threshold(self, new_threshold: float) -> None:
        """
        Adapt the decision threshold at runtime.

        Parameters
        ----------
        new_threshold : float
            New threshold value in [0, 1].
        """
        self.threshold = float(np.clip(new_threshold, 0.0, 1.0))
        logger.info("Auth threshold updated to %.3f.", self.threshold)
