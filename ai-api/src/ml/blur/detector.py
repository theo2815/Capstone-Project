from __future__ import annotations

import cv2
import numpy as np


class BlurDetector:
    """Detect image blur using Laplacian variance and FFT spectral analysis."""

    def __init__(self, laplacian_threshold: float = 100.0) -> None:
        self.laplacian_threshold = laplacian_threshold

    def detect(self, image: np.ndarray) -> dict:
        """Analyze an image for blur.

        Args:
            image: BGR numpy array from cv2.

        Returns:
            Dict with is_blurry, confidence, and metrics.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Laplacian variance: low variance = blurry
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # FFT-based: ratio of high-frequency energy to total energy
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8
        mask = np.ones_like(magnitude)
        mask[cy - r : cy + r, cx - r : cx + r] = 0
        total = np.sum(magnitude)
        hf_ratio = float(np.sum(magnitude * mask) / total) if total > 0 else 0.0

        is_blurry = laplacian_var < self.laplacian_threshold
        if is_blurry:
            confidence = 1.0 - min(1.0, laplacian_var / self.laplacian_threshold)
        else:
            confidence = min(1.0, laplacian_var / self.laplacian_threshold)

        return {
            "is_blurry": is_blurry,
            "confidence": confidence,
            "laplacian_variance": laplacian_var,
            "hf_ratio": hf_ratio,
        }
