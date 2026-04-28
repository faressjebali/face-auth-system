"""
Deep learning face recognition via the DeepFace library.

Supports FaceNet and ArcFace backends.  The module exposes:

* :class:`DeepFaceModel` – embedding extraction + cosine similarity comparison.
* :func:`cosine_similarity` – standalone helper.

DeepFace performs its own internal detection; set ``enforce_detection=False``
when passing pre-cropped patches from the preprocessing pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

import config

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Parameters
    ----------
    a, b : np.ndarray
        Embedding vectors of equal length.

    Returns
    -------
    float
        Similarity in [0, 1]; 1 = identical direction.
    """
    sim = 1.0 - float(cosine_distance(a.flatten(), b.flatten()))
    return float(np.clip(sim, 0.0, 1.0))


class DeepFaceModel:
    """
    Wrapper around the DeepFace library for embedding extraction and matching.

    Parameters
    ----------
    backend : str
        DeepFace model name; one of ``"Facenet"`` or ``"ArcFace"``.

    Attributes
    ----------
    backend : str
        Active DeepFace backend.
    distance_threshold : float
        Cosine *distance* below which two embeddings are considered the same
        identity (lower distance = higher similarity).
    """

    def __init__(
        self,
        backend: str = config.FACENET_BACKEND,
        distance_threshold: float = config.DEEP_DISTANCE_THRESHOLD,
    ) -> None:
        if backend not in (config.FACENET_BACKEND, config.ARCFACE_BACKEND):
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Choose '{config.FACENET_BACKEND}' or '{config.ARCFACE_BACKEND}'."
            )
        self.backend = backend
        self.distance_threshold = distance_threshold
        logger.info("DeepFaceModel initialised with backend '%s'.", backend)

    # ------------------------------------------------------------------
    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a face embedding vector from an image.

        Parameters
        ----------
        face_img : np.ndarray
            BGR face crop (uint8).  Pre-cropped patches should be passed with
            ``enforce_detection=False`` (handled internally).

        Returns
        -------
        Optional[np.ndarray]
            1-D float32 embedding vector, or ``None`` on failure.
        """
        try:
            from deepface import DeepFace  # deferred import (heavy)

            result = DeepFace.represent(
                img_path=face_img,
                model_name=self.backend,
                enforce_detection=False,
                detector_backend="skip",
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            # L2-normalise for cosine comparisons
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            return embedding
        except Exception as exc:
            logger.error(
                "Embedding extraction failed (backend=%s): %s", self.backend, exc
            )
            return None

    # ------------------------------------------------------------------
    def compare(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Compare two embeddings and decide whether they are the same identity.

        Parameters
        ----------
        embedding_a, embedding_b : np.ndarray
            L2-normalised embedding vectors.

        Returns
        -------
        Tuple[bool, float]
            ``(is_match, similarity_score)`` where *similarity_score* ∈ [0, 1].
        """
        sim = cosine_similarity(embedding_a, embedding_b)
        dist = 1.0 - sim
        is_match = dist < self.distance_threshold
        return is_match, sim

    # ------------------------------------------------------------------
    def identify(
        self,
        face_img: np.ndarray,
        known_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[str, float]:
        """
        Identify the closest matching identity from a gallery.

        Parameters
        ----------
        face_img : np.ndarray
            BGR face crop to identify.
        known_embeddings : Dict[str, np.ndarray]
            Mapping of ``user_id → stored_embedding``.

        Returns
        -------
        Tuple[str, float]
            ``(best_label, similarity_score)``.  Returns
            ``("unknown", 0.0)`` when no gallery is provided or embedding
            extraction fails.
        """
        if not known_embeddings:
            return "unknown", 0.0

        query_emb = self.extract_embedding(face_img)
        if query_emb is None:
            return "unknown", 0.0

        best_label = "unknown"
        best_sim = -1.0
        for uid, stored_emb in known_embeddings.items():
            try:
                sim = cosine_similarity(query_emb, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_label = uid
            except Exception as exc:
                logger.warning("Comparison error for user '%s': %s", uid, exc)

        dist = 1.0 - best_sim
        if dist >= self.distance_threshold:
            return "unknown", float(np.clip(best_sim, 0.0, 1.0))
        return best_label, float(np.clip(best_sim, 0.0, 1.0))

    # ------------------------------------------------------------------
    def batch_extract(
        self,
        face_imgs: List[np.ndarray],
    ) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for a list of face images.

        Parameters
        ----------
        face_imgs : List[np.ndarray]
            List of BGR face crops.

        Returns
        -------
        List[Optional[np.ndarray]]
            One embedding per input image; ``None`` on per-image failure.
        """
        return [self.extract_embedding(img) for img in face_imgs]
