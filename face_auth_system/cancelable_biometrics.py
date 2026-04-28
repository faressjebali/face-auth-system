"""
Cancelable biometrics via random orthogonal projection (BioHashing).

A secret *token* (e.g. derived from a PIN or smart card) seeds a random
orthogonal matrix that projects a full-dimensional embedding into a binary
protected template.  The template is:

  - **Non-invertible**: the projection matrix cannot be reversed without
    the token, protecting the underlying biometric.
  - **Revocable**: issuing a new token yields an uncorrelated template,
    so a compromised template can be invalidated.
  - **Diversity**: different tokens → different templates from the same
    biometric, preventing cross-system linking.

The approach follows the BioHashing paradigm (Jin et al., 2004).
"""

import hashlib
import logging
from typing import Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


def _seed_from_token(token: str) -> int:
    """Derive a reproducible 32-bit integer seed from a secret token."""
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _orthogonal_matrix(
    seed: int,
    input_dim: int,
    output_dim: int,
) -> np.ndarray:
    """
    Generate a stable random orthonormal projection matrix.

    Uses QR decomposition of a Gaussian random matrix so that rows form
    an orthonormal basis, preserving inner-product structure.

    Returns
    -------
    np.ndarray
        Shape ``(output_dim, input_dim)``.
    """
    rng = np.random.RandomState(seed)
    G = rng.randn(input_dim, output_dim).astype(np.float32)
    Q, _ = np.linalg.qr(G)             # Q: (input_dim, output_dim) orthonormal cols
    return Q[:, :output_dim].T          # (output_dim, input_dim)


class CancelableBiometrics:
    """
    Per-user cancelable template generator and comparator.

    Parameters
    ----------
    input_dim : int
        Size of raw embeddings (512 for FaceNet/ArcFace).
    output_dim : int
        Size of the protected template (256 by default).
    """

    def __init__(
        self,
        input_dim: int = config.PROJECTION_INPUT_DIM,
        output_dim: int = config.PROJECTION_OUTPUT_DIM,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._cache: dict = {}   # token → projection matrix

    # ------------------------------------------------------------------
    def _matrix(self, token: str) -> np.ndarray:
        if token not in self._cache:
            seed = _seed_from_token(token)
            self._cache[token] = _orthogonal_matrix(seed, self.input_dim, self.output_dim)
        return self._cache[token]

    # ------------------------------------------------------------------
    def protect(self, embedding: np.ndarray, token: str) -> np.ndarray:
        """
        Project *embedding* into the token-specific protected space and binarise.

        Parameters
        ----------
        embedding : np.ndarray
            1-D float32 embedding of length ``input_dim``.
        token : str
            User's secret token.

        Returns
        -------
        np.ndarray
            Binary float32 template of length ``output_dim``.
        """
        emb = embedding.flatten().astype(np.float32)
        if emb.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected embedding size {self.input_dim}, got {emb.shape[0]}. "
                "Pad or truncate before calling protect()."
            )
        projected = self._matrix(token) @ emb      # (output_dim,)
        return (projected >= 0).astype(np.float32)  # sign binarisation

    # ------------------------------------------------------------------
    def compare(self, template_a: np.ndarray, template_b: np.ndarray) -> float:
        """
        Normalised Hamming similarity between two binary protected templates.

        Returns
        -------
        float
            Similarity in [0, 1]; 1.0 = identical.
        """
        a, b = template_a.flatten(), template_b.flatten()
        if a.shape != b.shape:
            raise ValueError("Template shape mismatch.")
        return float(np.mean(a == b))

    # ------------------------------------------------------------------
    def revoke(self, old_token: str) -> None:
        """Evict the cached projection for *old_token* (template revocation)."""
        self._cache.pop(old_token, None)
        logger.info("Projection for old token evicted; re-enroll with a new token.")
