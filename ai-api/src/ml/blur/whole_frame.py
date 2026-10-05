"""Whole-frame measurements for the two-stage blur detector.

Stage 1 answers "is ANY part of this frame sharp?" with two numbers measured on one
640-px gray copy of the upload. Photos with no sharp region are whole-frame soft or
motion-smeared and never reach the neural network. The thresholds are learned from
the labelled photos by scripts/fit_blur_whole_frame_rule.py and live in src/config.py.
"""

from __future__ import annotations

import cv2
import numpy as np

MEASURE_DIM = 640  # same scale as the classical detector (BLUR_DETECT_DECODE_DIM)


def measure(image: np.ndarray) -> tuple[float, float]:
    """Return (fine_detail, direction_ratio) for a BGR or gray image.

    fine_detail: Laplacian variance at 640 px, the same score BlurDetector.detect_fast
    reports. direction_ratio: up-down detail energy over left-right detail energy; a
    sideways smear wipes out left-right detail, so the ratio climbs well above 1.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) > MEASURE_DIM:
        scale = MEASURE_DIM / max(h, w)
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    fine_detail = float(cv2.Laplacian(gray, cv2.CV_64F).var()) * MEASURE_DIM / max(gray.shape)
    g = gray.astype(np.float32)
    left_right = float((cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2).mean())
    up_down = float((cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2).mean())
    return fine_detail, up_down / (left_right + 1e-9)


def decide(fine_detail: float, direction_ratio: float, soft_max_detail: float,
           motion_min_ratio: float, motion_max_detail: float) -> str | None:
    """'motion_blurred' or 'defocused_blurred' when nothing is sharp, else None."""
    if direction_ratio > motion_min_ratio:
        return "motion_blurred" if fine_detail <= motion_max_detail else None
    return "defocused_blurred" if fine_detail <= soft_max_detail else None


def square_gray_view(image: np.ndarray, size: int) -> np.ndarray:
    """The whole frame as a size x size gray square: black bars, never a crop."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = gray
    # Area-averaging when shrinking: point-sampling would fake sharp edges (2026-08-14 ADR).
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interpolation)
