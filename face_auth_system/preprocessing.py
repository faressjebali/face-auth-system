"""
Face preprocessing module.

Provides face detection via OpenCV Haar cascades *and* Dlib HOG, 68-point
landmark-based affine alignment, and image normalization (resize, histogram
equalisation, grayscale + colour variants).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import dlib
import numpy as np

import config
from lighting import normalize_lighting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton detectors (loaded once per process)
# ---------------------------------------------------------------------------
_haar: Optional[cv2.CascadeClassifier] = None
_hog: Optional[dlib.fhog_object_detector] = None
_predictor: Optional[dlib.shape_predictor] = None


def _init_detectors() -> None:
    """Lazy-initialise all detector singletons."""
    global _haar, _hog, _predictor

    if _haar is None:
        cascade_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar = cv2.CascadeClassifier(cascade_xml)
        if _haar.empty():
            raise RuntimeError(f"Haar cascade not found at {cascade_xml}")

    if _hog is None:
        _hog = dlib.get_frontal_face_detector()

    if _predictor is None:
        model_path = config.DLIB_LANDMARK_MODEL
        if Path(model_path).exists():
            _predictor = dlib.shape_predictor(str(model_path))
        else:
            logger.warning(
                "Dlib landmark model missing at %s – alignment disabled. "
                "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                model_path,
            )


# ---------------------------------------------------------------------------
# Face Detection
# ---------------------------------------------------------------------------


def detect_faces_haar(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces with OpenCV Haar cascades.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale uint8 image.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Bounding boxes as ``(x, y, w, h)``.
    """
    _init_detectors()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detections = _haar.detectMultiScale(
        gray,
        scaleFactor=config.FACE_SCALE_FACTOR,
        minNeighbors=config.FACE_MIN_NEIGHBORS,
        minSize=config.FACE_MIN_SIZE,
    )
    if len(detections) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]


def detect_faces_dlib(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces with Dlib's HOG + SVM detector.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Bounding boxes as ``(x, y, w, h)``.
    """
    _init_detectors()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    rects = _hog(rgb, 1)
    return [(r.left(), r.top(), r.width(), r.height()) for r in rects]


def detect_faces(
    image: np.ndarray,
    method: str = "ensemble",
) -> List[Tuple[int, int, int, int]]:
    """
    Unified face detection supporting Haar, Dlib, or an ensemble.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    method : str
        ``'haar'``, ``'dlib'``, or ``'ensemble'`` (default).

    Returns
    -------
    List[Tuple[int, int, int, int]]
        NMS-filtered bounding boxes as ``(x, y, w, h)``.
    """
    boxes: List[Tuple[int, int, int, int]] = []

    if method in ("haar", "ensemble"):
        try:
            boxes.extend(detect_faces_haar(image))
        except Exception as exc:
            logger.error("Haar detection error: %s", exc)

    if method in ("dlib", "ensemble"):
        try:
            boxes.extend(detect_faces_dlib(image))
        except Exception as exc:
            logger.error("Dlib detection error: %s", exc)

    return _nms(boxes, iou_threshold=0.5)


def _nms(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Non-maximum suppression to deduplicate overlapping boxes."""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return [boxes[k] for k in keep]


# ---------------------------------------------------------------------------
# Landmark Detection
# ---------------------------------------------------------------------------


def get_landmarks(
    image: np.ndarray,
    rect: dlib.rectangle,
) -> Optional[np.ndarray]:
    """
    Extract 68 facial landmarks for a detected face rectangle.

    Parameters
    ----------
    image : np.ndarray
        BGR image containing the face.
    rect : dlib.rectangle
        Dlib bounding rectangle from the HOG detector.

    Returns
    -------
    Optional[np.ndarray]
        Array of shape ``(68, 2)`` with ``(x, y)`` pixel coordinates,
        or ``None`` when the landmark model is unavailable.
    """
    _init_detectors()
    if _predictor is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    shape = _predictor(rgb, rect)
    return np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(68)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Affine Alignment
# ---------------------------------------------------------------------------

# Dlib 68-point landmark indices for each eye
_LEFT_EYE_IDX = list(range(36, 42))
_RIGHT_EYE_IDX = list(range(42, 48))


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> np.ndarray:
    """
    Align a face using an affine transform derived from eye centre positions.

    Parameters
    ----------
    image : np.ndarray
        Full BGR image (not the crop).
    landmarks : np.ndarray
        68-point landmark array of shape ``(68, 2)``.
    output_size : Tuple[int, int]
        ``(width, height)`` of the returned aligned patch.

    Returns
    -------
    np.ndarray
        Aligned face patch of shape ``(height, width, channels)`` or
        ``(height, width)`` for grayscale input.
    """
    left_eye = landmarks[_LEFT_EYE_IDX].mean(axis=0)
    right_eye = landmarks[_RIGHT_EYE_IDX].mean(axis=0)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    desired_left = np.array([0.35 * output_size[0], 0.35 * output_size[1]])
    desired_right = np.array([0.65 * output_size[0], 0.35 * output_size[1]])
    desired_dist = float(np.linalg.norm(desired_right - desired_left))
    current_dist = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
    scale = desired_dist / current_dist

    eyes_center = ((left_eye + right_eye) / 2).astype(np.float32)
    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
    M[0, 2] += output_size[0] * 0.5 - eyes_center[0]
    M[1, 2] += output_size[1] * 0.35 - eyes_center[1]

    return cv2.warpAffine(image, M, output_size, flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_face(
    face_img: np.ndarray,
    output_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> dict:
    """
    Resize, equalise histogram, and produce grayscale + colour variants.

    Parameters
    ----------
    face_img : np.ndarray
        BGR or grayscale face crop.
    output_size : Tuple[int, int]
        ``(width, height)`` to resize to.

    Returns
    -------
    dict
        ``gray``      – uint8 (H, W) histogram-equalised grayscale.
        ``color``     – uint8 (H, W, 3) resized BGR.
        ``gray_norm`` – float32 (H, W) in [0, 1].
    """
    resized = cv2.resize(face_img, output_size, interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        gray = resized
        color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    else:
        color = normalize_lighting(resized)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    gray_eq = cv2.equalizeHist(gray)
    return {
        "gray": gray_eq,
        "color": color,
        "gray_norm": gray_eq.astype(np.float32) / 255.0,
    }


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


def preprocess_image(
    image: np.ndarray,
    output_size: Tuple[int, int] = config.IMAGE_SIZE,
    align: bool = True,
    method: str = "ensemble",
) -> List[dict]:
    """
    End-to-end preprocessing: detect → align → normalise.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    output_size : Tuple[int, int]
        Target ``(width, height)`` for the output face patches.
    align : bool
        Apply landmark-based alignment when possible.
    method : str
        Detection method; see :func:`detect_faces`.

    Returns
    -------
    List[dict]
        One ``dict`` per detected face (keys from :func:`normalize_face`).
        Empty list when no faces are detected.
    """
    results: List[dict] = []
    try:
        boxes = detect_faces(image, method=method)
        if not boxes:
            logger.debug("No faces detected in image.")
            return results

        for x, y, w, h in boxes:
            face_src = image  # alignment works on the full image
            processed = image[y: y + h, x: x + w]  # fallback crop

            if align and _predictor is not None:
                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = get_landmarks(image, dlib_rect)
                if landmarks is not None:
                    try:
                        processed = align_face(image, landmarks, output_size)
                    except Exception as exc:
                        logger.warning("Alignment failed, falling back to crop: %s", exc)

            results.append(normalize_face(processed, output_size))

    except Exception as exc:
        logger.error("preprocess_image pipeline error: %s", exc)

    return results
