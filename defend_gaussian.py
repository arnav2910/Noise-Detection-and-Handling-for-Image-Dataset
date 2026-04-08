"""
defend_gaussian.py
------------------
Removes Gaussian (random) noise from images.
Strategy: Gaussian blur + optional Non-Local Means denoising.
"""

import numpy as np
import cv2


def defend(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """
    Remove Gaussian noise.

    Parameters
    ----------
    img      : BGR or grayscale uint8 image
    strength : "light" | "medium" | "strong"

    Returns
    -------
    Cleaned image (same shape/dtype as input)
    """
    params = {
        "light":  {"ksize": (3, 3), "sigma": 0.8, "nlm_h": 5},
        "medium": {"ksize": (5, 5), "sigma": 1.2, "nlm_h": 10},
        "strong": {"ksize": (7, 7), "sigma": 1.8, "nlm_h": 15},
    }
    p = params.get(strength, params["medium"])

    # Step 1 – Gaussian blur removes bulk of random noise
    blurred = cv2.GaussianBlur(img, p["ksize"], p["sigma"])

    # Step 2 – Non-Local Means further suppresses residual noise
    if img.ndim == 3:
        cleaned = cv2.fastNlMeansDenoisingColored(
            blurred,
            None,
            h=p["nlm_h"],
            hColor=p["nlm_h"],
            templateWindowSize=7,
            searchWindowSize=21,
        )
    else:
        cleaned = cv2.fastNlMeansDenoising(
            blurred,
            None,
            h=p["nlm_h"],
            templateWindowSize=7,
            searchWindowSize=21,
        )

    return cleaned
