"""
defend_adversarial.py
---------------------
Defends against adversarial perturbations (FGSM, PGD-style attacks).

Strategy (ensemble of three methods):
  1. JPEG compression  – destroys high-freq adversarial signal
  2. Bit-depth reduction – quantises subtle pixel differences away
  3. Gaussian blur      – low-pass filter removes fine perturbations

All three are applied and the result is a soft combination that
maximally suppresses the adversarial component while minimising
semantic degradation.
"""

import io
import numpy as np
import cv2
from PIL import Image


def _jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Encode to JPEG in memory then decode – destroys adversarial noise."""
    is_gray = img.ndim == 2
    pil_img = Image.fromarray(img if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf))
    if not is_gray:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def _bit_depth_reduce(img: np.ndarray, bits: int = 5) -> np.ndarray:
    """Reduce effective bit depth to crush tiny perturbations."""
    factor = 2 ** (8 - bits)
    return ((img.astype(np.float32) // factor) * factor).astype(np.uint8)


def defend(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """
    Remove adversarial perturbations using an ensemble approach.

    Parameters
    ----------
    img      : BGR or grayscale uint8 image
    strength : "light" | "medium" | "strong"

    Returns
    -------
    Cleaned image (same shape/dtype as input)
    """
    params = {
        "light":  {"jpeg_q": 85, "bits": 7, "blur_k": 3, "blur_s": 0.5},
        "medium": {"jpeg_q": 70, "bits": 6, "blur_k": 3, "blur_s": 0.8},
        "strong": {"jpeg_q": 55, "bits": 5, "blur_k": 5, "blur_s": 1.0},
    }
    p = params.get(strength, params["medium"])

    # Method 1: JPEG compression
    jpeg_out = _jpeg_compress(img, p["jpeg_q"])

    # Method 2: Bit-depth reduction
    bits_out = _bit_depth_reduce(img, p["bits"])

    # Method 3: Gaussian blur on JPEG output (compound)
    blur_out = cv2.GaussianBlur(jpeg_out, (p["blur_k"], p["blur_k"]), p["blur_s"])

    # Ensemble average – weighted towards JPEG (most effective)
    combined = (
        0.5  * jpeg_out.astype(np.float32) +
        0.25 * bits_out.astype(np.float32) +
        0.25 * blur_out.astype(np.float32)
    )

    return np.clip(combined, 0, 255).astype(np.uint8)
