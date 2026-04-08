"""
noise_detector.py
-----------------
Detects the type of noise present in an image.
Supports: Gaussian, Salt & Pepper, Blur, Compression Artifacts,
          Adversarial Perturbation, Clean

Features extracted
------------------
pixel_variance       : overall pixel intensity spread
laplacian_variance   : edge sharpness — low = blurry
salt_pepper_ratio    : fraction of pixels exactly 0 or 255
high_freq_energy     : FFT energy in outer ring
block_artifact_score : discontinuity at every 8-pixel JPEG boundary
smooth_region_noise  : mean residual in very smooth (grad<3) regions
smooth_region_p90    : 90th-percentile residual in smooth regions
                       (adversarial key: p90 > 0.7 + mean > 0.3)
"""

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class DetectionResult:
    noise_type: str
    confidence: float
    details: dict


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _variance(gray: np.ndarray) -> float:
    return float(np.var(gray))


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _salt_pepper_ratio(gray: np.ndarray) -> float:
    return float(np.sum((gray == 0) | (gray == 255))) / gray.size


def _high_freq_energy(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    outer_energy = magnitude[radius > min(cy, cx) * 0.5].sum()
    return float(outer_energy / (magnitude.sum() + 1e-8))


def _block_artifact_score(gray: np.ndarray) -> float:
    scores = []
    g = gray.astype(np.float64)
    for i in range(8, gray.shape[0], 8):
        scores.append(np.abs(g[i, :] - g[i - 1, :]).mean())
    for j in range(8, gray.shape[1], 8):
        scores.append(np.abs(g[:, j] - g[:, j - 1]).mean())
    return float(np.mean(scores)) if scores else 0.0


def _smooth_region_noise(gray: np.ndarray):
    """
    Adversarial perturbations create tiny but UNIFORM residuals in the
    smoothest image regions (gradient magnitude < 3).

    Clean images:          mean < 0.20,  p90 < 0.30
    Adversarially noised:  mean > 0.35,  p90 > 0.70

    Uses a 3x3 local mean window (tighter than 5x5) to capture
    pixel-level perturbations before they blur away.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    smooth_mask = np.sqrt(gx ** 2 + gy ** 2) < 3

    if smooth_mask.sum() < 50:
        return 0.0, 0.0

    local_mean = cv2.blur(g, (3, 3))
    residual = np.abs(g - local_mean)
    masked = residual[smooth_mask]
    return float(masked.mean()), float(np.percentile(masked, 90))


def detect(img: np.ndarray) -> DetectionResult:
    """
    Analyse a single image and return the most likely noise type.

    Parameters
    ----------
    img : np.ndarray  BGR or grayscale uint8 image.
    Returns DetectionResult
    """
    gray = _to_gray(img)

    var          = _variance(gray)
    lap_var      = _laplacian_variance(gray)
    sp_ratio     = _salt_pepper_ratio(gray)
    hf_energy    = _high_freq_energy(gray)
    block_score  = _block_artifact_score(gray)
    smooth_mean, smooth_p90 = _smooth_region_noise(gray)

    details = {
        "pixel_variance":       round(var, 2),
        "laplacian_variance":   round(lap_var, 2),
        "salt_pepper_ratio":    round(sp_ratio, 5),
        "high_freq_energy":     round(hf_energy, 4),
        "block_artifact_score": round(block_score, 4),
        "smooth_noise_mean":    round(smooth_mean, 4),
        "smooth_noise_p90":     round(smooth_p90, 4),
    }

    # ── Decision tree ──────────────────────────────────────────────────────

    # 1. Salt & Pepper
    #    Real photos legitimately have <1% extreme pixels.
    #    Artificially noised: 3–10%. Threshold at 2.5% catches all real cases.
    if sp_ratio > 0.025:
        confidence = min(1.0, sp_ratio / 0.025 * 0.5 + 0.5)
        return DetectionResult("salt_pepper", round(confidence, 2), details)

    # 2. Gaussian
    #    Heavy Gaussian noise (σ≥50): var > 2000 AND block_score > 30 AND
    #    lap_var > 5000 all fire simultaneously. Natural images don't hit all three.
    if var > 2000 and block_score > 30 and lap_var > 5000:
        confidence = min(1.0, (block_score - 30) / 30 + 0.5)
        return DetectionResult("gaussian", round(confidence, 2), details)

    # 3. Blur
    #    Near-zero Laplacian = severe blur (not artistic bokeh which scores 100+).
    if lap_var < 5:
        confidence = max(0.6, 1.0 - lap_var / 10)
        return DetectionResult("blur", round(confidence, 2), details)

    # 4. Adversarial perturbation
    #    Signature: BOTH mean residual AND p90 residual elevated in smooth regions.
    #    Requiring both prevents false positives from soft/compressed clean images.
    #    Calibrated on real images:
    #      Clean (Panda 1):       mean=0.067  p90=0.111
    #      Clean (Balatro):       mean=0.092  p90=0.111
    #      Clean (portrait):      mean=0.191  p90=0.667
    #      Adversarial (Panda 2): mean=0.484  p90=0.889
    if smooth_mean > 0.35 and smooth_p90 > 0.75 and sp_ratio < 0.01:
        confidence = min(1.0, (smooth_mean - 0.35) / 0.3 + 0.55)
        return DetectionResult("adversarial", round(confidence, 2), details)

    # 5. Compression
    if block_score > 20.0:
        confidence = min(1.0, block_score / 40)
        return DetectionResult("compression", round(confidence, 2), details)

    # 6. Clean
    return DetectionResult("clean", 0.95, details)
