"""Lighting normalization utilities — pure OpenCV, no dlib dependency."""

import cv2
import numpy as np


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


def normalize_lighting(img: np.ndarray) -> np.ndarray:
    """
    Adaptive gamma correction + CLAHE on the LAB L-channel.

    Handles global brightness shifts (dark room, backlighting) and local
    illumination gradients (side lighting, partial shadow) without altering
    colour hue or face geometry.

    Parameters
    ----------
    img : np.ndarray
        BGR uint8 image.

    Returns
    -------
    np.ndarray
        Lighting-normalised BGR uint8 image, same shape as *img*.
        Returned unchanged when *img* is not a 3-channel colour image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    # Adaptive gamma: steer mean luminance toward 128
    gray_mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    gamma = float(np.clip(
        np.log(128.0) / np.log(max(gray_mean, 1.0)),
        0.4, 3.0,
    ))
    img = _apply_gamma(img, gamma)

    # CLAHE on the L channel to equalise local contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
