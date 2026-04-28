"""
Classical face recognition models.

Implements:
* **EigenfacesModel** – PCA projection + nearest-neighbour classifier
  (scikit-learn).
* **LBPHModel** – Local Binary Pattern Histograms via
  ``cv2.face.LBPHFaceRecognizer``.

Both expose a common interface: ``train``, ``predict``, ``save``, ``load``.
``predict`` always returns ``(label: str, score: float)`` where *score* is
normalised to [0, 1] and higher means more confident.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eigenfaces (PCA + k-NN)
# ---------------------------------------------------------------------------


class EigenfacesModel:
    """
    Eigenfaces face recogniser built on PCA + k-NN.

    Attributes
    ----------
    n_components : int
        Number of PCA components to retain.
    pipeline : sklearn.pipeline.Pipeline or None
        Fitted pipeline (PCA + StandardScaler + k-NN).
    label_encoder : sklearn.preprocessing.LabelEncoder
        Maps integer class indices ↔ string labels.
    image_shape : Tuple[int, int]
        Spatial shape of the training face images.
    """

    def __init__(self, n_components: int = config.PCA_N_COMPONENTS) -> None:
        self.n_components = n_components
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.image_shape: Tuple[int, int] = config.IMAGE_SIZE

    # ------------------------------------------------------------------
    def train(self, faces: List[np.ndarray], labels: List[str]) -> None:
        """
        Fit the PCA projection and k-NN classifier.

        Parameters
        ----------
        faces : List[np.ndarray]
            Grayscale face images of uniform shape.
        labels : List[str]
            Corresponding user/class identifiers.

        Raises
        ------
        ValueError
            If fewer unique classes than ``n_components`` are provided.
        """
        if len(faces) == 0:
            raise ValueError("No training faces provided.")

        self.image_shape = faces[0].shape[:2]
        X = np.array([f.flatten().astype(np.float32) for f in faces])
        y = self.label_encoder.fit_transform(labels)

        n_components = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pipeline = Pipeline([
            ("pca", PCA(n_components=n_components, whiten=True)),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=1, metric="euclidean")),
        ])
        self.pipeline.fit(X, y)
        logger.info(
            "Eigenfaces trained on %d samples, %d classes, %d components.",
            len(faces), len(set(labels)), n_components,
        )

    # ------------------------------------------------------------------
    def predict(self, face: np.ndarray) -> Tuple[str, float]:
        """
        Predict the identity of a face image.

        Parameters
        ----------
        face : np.ndarray
            Grayscale face image (same shape as training images).

        Returns
        -------
        Tuple[str, float]
            ``(label, score)`` where *score* ∈ [0, 1]; 1 = perfect match.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if self.pipeline is None:
            raise RuntimeError("EigenfacesModel has not been trained.")

        x = face.flatten().astype(np.float32).reshape(1, -1)
        pred_idx = int(self.pipeline.predict(x)[0])
        label = str(self.label_encoder.inverse_transform([pred_idx])[0])

        # Distance to the nearest training neighbour → normalised score
        knn: KNeighborsClassifier = self.pipeline.named_steps["knn"]
        pca: PCA = self.pipeline.named_steps["pca"]
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        x_proj = scaler.transform(pca.transform(x))
        dist, _ = knn.kneighbors(x_proj, n_neighbors=1)
        raw_dist = float(dist[0, 0])
        # Convert distance to similarity score; clip for robustness
        score = float(np.clip(1.0 / (1.0 + raw_dist / 10.0), 0.0, 1.0))
        return label, score

    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Serialise the model to disk with pickle.

        Parameters
        ----------
        path : Path, optional
            Destination file.  Defaults to ``config.EIGENFACES_MODEL_PATH``.

        Returns
        -------
        Path
            Path where the model was saved.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained model.")
        save_path = Path(path or config.EIGENFACES_MODEL_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(
                {
                    "pipeline": self.pipeline,
                    "label_encoder": self.label_encoder,
                    "image_shape": self.image_shape,
                    "n_components": self.n_components,
                },
                fh,
            )
        logger.info("EigenfacesModel saved to %s.", save_path)
        return save_path

    # ------------------------------------------------------------------
    def load(self, path: Optional[Path] = None) -> None:
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        path : Path, optional
            Source file.  Defaults to ``config.EIGENFACES_MODEL_PATH``.
        """
        load_path = Path(path or config.EIGENFACES_MODEL_PATH)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        with open(load_path, "rb") as fh:
            data = pickle.load(fh)
        self.pipeline = data["pipeline"]
        self.label_encoder = data["label_encoder"]
        self.image_shape = data["image_shape"]
        self.n_components = data["n_components"]
        logger.info("EigenfacesModel loaded from %s.", load_path)


# ---------------------------------------------------------------------------
# LBPH Recogniser
# ---------------------------------------------------------------------------


class LBPHModel:
    """
    Local Binary Pattern Histograms (LBPH) face recogniser.

    Wraps OpenCV's ``LBPHFaceRecognizer`` and adds a label-string mapping
    layer on top of its integer-label interface.

    Attributes
    ----------
    recognizer : cv2.face.LBPHFaceRecognizer
        Underlying OpenCV LBPH recogniser.
    label_map : Dict[int, str]
        Maps integer label → user identifier string.
    reverse_map : Dict[str, int]
        Maps user identifier string → integer label.
    """

    def __init__(
        self,
        radius: int = config.LBPH_RADIUS,
        neighbors: int = config.LBPH_NEIGHBORS,
        grid_x: int = config.LBPH_GRID_X,
        grid_y: int = config.LBPH_GRID_Y,
    ) -> None:
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=radius,
            neighbors=neighbors,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        self.label_map: Dict[int, str] = {}
        self.reverse_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def train(self, faces: List[np.ndarray], labels: List[str]) -> None:
        """
        Train the LBPH recogniser.

        Parameters
        ----------
        faces : List[np.ndarray]
            Grayscale face images (uint8).
        labels : List[str]
            Corresponding user identifiers.
        """
        if len(faces) == 0:
            raise ValueError("No training faces provided.")

        unique_labels = sorted(set(labels))
        self.label_map = {i: lbl for i, lbl in enumerate(unique_labels)}
        self.reverse_map = {lbl: i for i, lbl in self.label_map.items()}

        int_labels = np.array([self.reverse_map[l] for l in labels], dtype=np.int32)
        gray_faces = [
            f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            for f in faces
        ]
        self.recognizer.train(gray_faces, int_labels)
        logger.info(
            "LBPHModel trained on %d samples, %d classes.",
            len(faces), len(unique_labels),
        )

    # ------------------------------------------------------------------
    def predict(self, face: np.ndarray) -> Tuple[str, float]:
        """
        Predict the identity of a face image.

        Parameters
        ----------
        face : np.ndarray
            Grayscale face image (uint8).

        Returns
        -------
        Tuple[str, float]
            ``(label, score)`` where *score* ∈ [0, 1]; higher = more confident.
        """
        if not self.label_map:
            raise RuntimeError("LBPHModel has not been trained.")

        gray = face if face.ndim == 2 else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        label_idx, raw_confidence = self.recognizer.predict(gray)
        label = self.label_map.get(int(label_idx), "unknown")

        # OpenCV LBPH: 0 = perfect match, ↑ = worse.  Normalise to [0, 1].
        score = float(
            np.clip(1.0 - raw_confidence / config.LBPH_MAX_CONFIDENCE, 0.0, 1.0)
        )
        return label, score

    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the LBPH model and label maps to disk.

        Parameters
        ----------
        path : Path, optional
            Directory or file stem.  Defaults to ``config.LBPH_MODEL_PATH``.

        Returns
        -------
        Path
            Path to the saved XML file.
        """
        save_path = Path(path or config.LBPH_MODEL_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.recognizer.save(str(save_path))
        # Persist label maps alongside the XML
        meta_path = save_path.with_suffix(".pkl")
        with open(meta_path, "wb") as fh:
            pickle.dump(
                {"label_map": self.label_map, "reverse_map": self.reverse_map},
                fh,
            )
        logger.info("LBPHModel saved to %s.", save_path)
        return save_path

    # ------------------------------------------------------------------
    def load(self, path: Optional[Path] = None) -> None:
        """
        Load a saved LBPH model from disk.

        Parameters
        ----------
        path : Path, optional
            Source XML file.  Defaults to ``config.LBPH_MODEL_PATH``.
        """
        load_path = Path(path or config.LBPH_MODEL_PATH)
        if not load_path.exists():
            raise FileNotFoundError(f"LBPH model not found: {load_path}")
        self.recognizer.read(str(load_path))
        meta_path = load_path.with_suffix(".pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as fh:
                meta = pickle.load(fh)
            self.label_map = meta["label_map"]
            self.reverse_map = meta["reverse_map"]
        logger.info("LBPHModel loaded from %s.", load_path)
