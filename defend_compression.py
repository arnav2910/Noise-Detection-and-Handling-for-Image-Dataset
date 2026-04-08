"""
defend_compression.py
---------------------
Removes JPEG/compression block artifacts.
Strategy: Bilateral filter (edge-preserving) + optional DCT-domain smoothing.
"""

import numpy as np
import cv2


def defend(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """
    Remove compression (JPEG block) artifacts.

    Parameters
    ----------
    img      : BGR or grayscale uint8 image
    strength : "light" | "medium" | "strong"

    Returns
    -------
    Cleaned image (same shape/dtype as input)
    """
    params = {
        "light":  {"d": 5,  "sigmaColor": 20,  "sigmaSpace": 20},
        "medium": {"d": 9,  "sigmaColor": 50,  "sigmaSpace": 50},
        "strong": {"d": 15, "sigmaColor": 100, "sigmaSpace": 100},
    }
    p = params.get(strength, params["medium"])

    # Bilateral filter: smooths flat regions, preserves real edges
    if img.ndim == 3:
        cleaned = cv2.bilateralFilter(img, p["d"], p["sigmaColor"], p["sigmaSpace"])
    else:
        cleaned = cv2.bilateralFilter(img, p["d"], p["sigmaColor"], p["sigmaSpace"])

    # Second pass: gentle Gaussian to further soften residual ringing
    if strength == "strong":
        cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0.5)

    return cleaned
