"""
defend_blur.py
--------------
Corrects blur degradation via Unsharp Masking + optional Laplacian sharpening.
Note: motion/defocus blur cannot be perfectly inverted without the blur kernel;
this approach is a practical enhancement, not perfect deconvolution.
"""

import numpy as np
import cv2


def defend(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """
    Sharpen a blurry image.

    Parameters
    ----------
    img      : BGR or grayscale uint8 image
    strength : "light" | "medium" | "strong"

    Returns
    -------
    Sharpened image (same shape/dtype as input)
    """
    params = {
        "light":  {"amount": 0.5, "radius": 1.0, "threshold": 10},
        "medium": {"amount": 1.0, "radius": 1.5, "threshold": 5},
        "strong": {"amount": 1.8, "radius": 2.0, "threshold": 0},
    }
    p = params.get(strength, params["medium"])

    img_f = img.astype(np.float32)

    # Gaussian blur to create the "mask"
    ksize = max(3, int(p["radius"] * 4) | 1)   # must be odd
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), p["radius"])

    # Unsharp mask: sharpened = original + amount * (original - blurred)
    mask = img_f - blurred
    sharpened = img_f + p["amount"] * mask

    # Threshold: only sharpen where difference is significant
    if p["threshold"] > 0:
        low_contrast = np.abs(mask) < p["threshold"]
        sharpened[low_contrast] = img_f[low_contrast]

    return np.clip(sharpened, 0, 255).astype(np.uint8)
