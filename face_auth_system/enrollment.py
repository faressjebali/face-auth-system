"""
User enrollment workflow.

Captures or loads face images, runs the preprocessing pipeline, extracts
deep embeddings, stores a cancelable protected template in the vault, and
saves raw face crops for classical model retraining.

Public API
----------
EnrollmentManager.enroll_from_images()   — enroll from file paths
EnrollmentManager.enroll_from_camera()   — interactive webcam enrollment
EnrollmentManager.delete_user()          — remove all data for a user
EnrollmentManager.list_enrolled()        — list enrolled user IDs
EnrollmentManager.train_classical_models() — retrain Eigenfaces + LBPH
"""

import logging
import pickle
import shutil
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from cancelable_biometrics import CancelableBiometrics
from classical_model import EigenfacesModel, LBPHModel
from deep_model import DeepFaceModel
from vault import EmbeddingVault

logger = logging.getLogger(__name__)

_GALLERY_PATH = config.MODELS_DIR / "deep_gallery.pkl"


# ---------------------------------------------------------------------------
# Gallery helpers (deep embedding gallery persisted as pickle)
# ---------------------------------------------------------------------------


def _load_gallery() -> Dict[str, np.ndarray]:
    if _GALLERY_PATH.exists():
        with open(_GALLERY_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_gallery(gallery: Dict[str, np.ndarray]) -> None:
    _GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_GALLERY_PATH, "wb") as f:
        pickle.dump(gallery, f)


# ---------------------------------------------------------------------------
# EnrollmentManager
# ---------------------------------------------------------------------------


class EnrollmentManager:
    """
    Orchestrates face enrollment and user management.

    Parameters
    ----------
    vault : EmbeddingVault, optional
    deep_model : DeepFaceModel, optional
    cancelable : CancelableBiometrics, optional
    """

    def __init__(
        self,
        vault: Optional[EmbeddingVault] = None,
        deep_model: Optional[DeepFaceModel] = None,
        cancelable: Optional[CancelableBiometrics] = None,
    ) -> None:
        self.vault = vault or EmbeddingVault()
        self.deep_model = deep_model or DeepFaceModel()
        self.cancelable = cancelable or CancelableBiometrics()
        self._gallery: Dict[str, np.ndarray] = _load_gallery()

    # ------------------------------------------------------------------
    def enroll_from_images(
        self,
        user_id: str,
        image_paths: List[str],
        password: str,
        token: str,
    ) -> bool:
        """
        Enroll a user from a list of image file paths.

        Parameters
        ----------
        user_id : str
            Unique user identifier.
        image_paths : List[str]
            Paths to face images (1-N per user; more = more robust).
        password : str
            Vault encryption password.
        token : str
            Secret token for cancelable biometrics.

        Returns
        -------
        bool
            ``True`` on success.
        """
        embeddings: List[np.ndarray] = []
        face_crops: List[np.ndarray] = []

        from preprocessing import preprocess_image  # lazy — dlib optional
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Cannot read image: %s", path)
                continue
            faces = preprocess_image(img)
            if not faces:
                logger.warning("No face detected in %s", path)
                continue
            face = faces[0]
            face_crops.append(face["color"])
            emb = self.deep_model.extract_embedding(face["color"])
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.error("No embeddings extracted for user '%s'.", user_id)
            return False

        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb /= norm

        # Pad / truncate to match the projection input dimension
        target = config.PROJECTION_INPUT_DIM
        if mean_emb.shape[0] < target:
            mean_emb = np.pad(mean_emb, (0, target - mean_emb.shape[0]))
        elif mean_emb.shape[0] > target:
            mean_emb = mean_emb[:target]

        protected = self.cancelable.protect(mean_emb, token)
        self.vault.store(user_id, protected, password)

        self._gallery[user_id] = mean_emb
        _save_gallery(self._gallery)

        # Save crops for classical model retraining
        user_dir = config.ENROLLED_FACES_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        for idx, crop in enumerate(face_crops):
            cv2.imwrite(str(user_dir / f"{idx:04d}.jpg"), crop)

        logger.info(
            "Enrolled '%s': %d images → %d embeddings.",
            user_id, len(image_paths), len(embeddings),
        )
        return True

    # ------------------------------------------------------------------
    def enroll_from_camera(
        self,
        user_id: str,
        password: str,
        token: str,
        n_captures: int = 10,
        camera_id: int = 0,
    ) -> bool:
        """
        Interactive webcam enrollment.

        Press **SPACE** to capture a frame, **Q** to abort.

        Parameters
        ----------
        n_captures : int
            Number of face frames to collect before enrolling.
        camera_id : int
            OpenCV camera device index.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Cannot open camera %d.", camera_id)
            return False

        frames: List[np.ndarray] = []
        logger.info(
            "Webcam enrollment for '%s'. SPACE=capture (%d needed)  Q=abort.",
            user_id, n_captures,
        )

        try:
            from preprocessing import preprocess_image  # lazy — dlib optional
            while len(frames) < n_captures:
                ret, frame = cap.read()
                if not ret:
                    break
                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"Captures: {len(frames)}/{n_captures}  [SPACE] capture  [Q] abort",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2,
                )
                cv2.imshow("Enrollment", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if preprocess_image(frame):
                        frames.append(frame.copy())
                        logger.debug("Captured frame %d.", len(frames))
                    else:
                        logger.warning("No face detected — try again.")
                elif key == ord("q"):
                    logger.info("Enrollment aborted by user.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if not frames:
            return False

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, f in enumerate(frames):
                p = os.path.join(tmp, f"cap_{i:04d}.jpg")
                cv2.imwrite(p, f)
                paths.append(p)
            return self.enroll_from_images(user_id, paths, password, token)

    # ------------------------------------------------------------------
    def delete_user(self, user_id: str) -> bool:
        """Remove all enrolled data for *user_id*."""
        removed = self.vault.delete(user_id)
        if user_id in self._gallery:
            del self._gallery[user_id]
            _save_gallery(self._gallery)
            removed = True
        user_dir = config.ENROLLED_FACES_DIR / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)
            removed = True
        if removed:
            logger.info("Deleted enrollment for '%s'.", user_id)
        return removed

    # ------------------------------------------------------------------
    def list_enrolled(self) -> List[str]:
        """Return user IDs present in the deep gallery."""
        return list(self._gallery.keys())

    # ------------------------------------------------------------------
    def train_classical_models(self) -> Tuple[bool, bool]:
        """
        Retrain Eigenfaces and LBPH from the enrolled face crops.

        Requires at least 2 distinct enrolled users.

        Returns
        -------
        Tuple[bool, bool]
            ``(eigenfaces_ok, lbph_ok)``
        """
        X: List[np.ndarray] = []
        labels: List[str] = []

        for user_dir in sorted(config.ENROLLED_FACES_DIR.iterdir()):
            if not user_dir.is_dir():
                continue
            for img_path in sorted(user_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(img)
                    labels.append(user_dir.name)

        if len(set(labels)) < 2:
            logger.warning("Need ≥2 enrolled users to train classical models.")
            return False, False

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        eigen_ok = lbph_ok = False

        try:
            ef = EigenfacesModel()
            ef.train(X, labels)
            ef.save(config.EIGENFACES_MODEL_PATH)
            eigen_ok = True
            logger.info("Eigenfaces trained (%d samples, %d users).", len(X), len(set(labels)))
        except Exception as exc:
            logger.error("Eigenfaces training error: %s", exc)

        try:
            lbph = LBPHModel()
            lbph.train(X, labels)
            lbph.save(config.LBPH_MODEL_PATH)
            lbph_ok = True
            logger.info("LBPH trained (%d samples, %d users).", len(X), len(set(labels)))
        except Exception as exc:
            logger.error("LBPH training error: %s", exc)

        return eigen_ok, lbph_ok
