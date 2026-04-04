"""
pipeline.py
-----------
Orchestrates: detect → dispatch to correct defender → return result.

Usage:
    from pipeline import process_image, process_batch

    result = process_image(img_array)
    # result.noise_type, result.cleaned_image, result.detection
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from detectors.noise_detector import detect, DetectionResult

# Defenders
from defenders import (
    defend_gaussian,
    defend_salt_pepper,
    defend_blur,
    defend_compression,
    defend_adversarial,
)

DEFENDER_MAP = {
    "gaussian":    defend_gaussian.defend,
    "salt_pepper": defend_salt_pepper.defend,
    "blur":        defend_blur.defend,
    "compression": defend_compression.defend,
    "adversarial": defend_adversarial.defend,
    "clean":       None,   # no action needed
}


@dataclass
class PipelineResult:
    noise_type: str
    confidence: float
    cleaned_image: np.ndarray
    was_modified: bool
    details: dict


def process_image(img: np.ndarray, strength: str = "medium") -> PipelineResult:
    """
    Full pipeline for a single image.

    Parameters
    ----------
    img      : BGR or grayscale uint8 numpy array
    strength : defence strength ("light" | "medium" | "strong")

    Returns
    -------
    PipelineResult
    """
    detection: DetectionResult = detect(img)

    defender = DEFENDER_MAP.get(detection.noise_type)

    if defender is not None:
        cleaned = defender(img, strength=strength)
        was_modified = True
    else:
        cleaned = img.copy()
        was_modified = False

    return PipelineResult(
        noise_type=detection.noise_type,
        confidence=detection.confidence,
        cleaned_image=cleaned,
        was_modified=was_modified,
        details=detection.details,
    )


def process_batch(images: list[tuple[str, np.ndarray]], strength: str = "medium"):
    """
    Process multiple images and return results + a summary report.

    Parameters
    ----------
    images   : list of (filename, img_array) tuples
    strength : defence strength

    Returns
    -------
    results : list of (filename, PipelineResult)
    report  : dict with counts per noise type
    """
    results = []
    report_counts = {
        "clean": 0,
        "gaussian": 0,
        "salt_pepper": 0,
        "blur": 0,
        "compression": 0,
        "adversarial": 0,
    }

    for filename, img in images:
        result = process_image(img, strength=strength)
        results.append((filename, result))
        report_counts[result.noise_type] = report_counts.get(result.noise_type, 0) + 1

    total = len(images)
    report = {
        "total_images": total,
        "counts": report_counts,
        "percentages": {
            k: round(v / total * 100, 1) if total > 0 else 0
            for k, v in report_counts.items()
        },
    }

    return results, report
