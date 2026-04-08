"""
defend_salt_pepper.py
---------------------
Removes salt-and-pepper (impulse) noise.
Strategy: Median filter (best for impulse noise; preserves edges).
"""

import numpy as np
import cv2


def defend(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """
    Remove salt-and-pepper noise.

    Parameters
    ----------
    img      : BGR or grayscale uint8 image
    strength : "light" | "medium" | "strong"

    Returns
    -------
    Cleaned image (same shape/dtype as input)
    """
    ksize = {"light": 3, "medium": 5, "strong": 7}.get(strength, 5)

    if img.ndim == 3:
        # Apply median per channel so colours don't bleed
        channels = [cv2.medianBlur(img[:, :, c], ksize) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    else:
        return cv2.medianBlur(img, ksize)
