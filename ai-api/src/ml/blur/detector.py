from __future__ import annotations

import cv2
import numpy as np

try:
    from _eventai_cpp import laplacian_variance as _cpp_laplacian_var
    from _eventai_cpp import fft_hf_ratio as _cpp_hf_ratio

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


class BlurDetector:
    """Fast coarse blur gate using Laplacian variance and FFT spectral analysis.

    This detector provides a binary is-blurry/is-sharp signal based on global
    frequency content. It works well for uniformly blurred images but cannot
    distinguish blur *types* (portrait defocus vs motion blur) or detect
    spatially varying blur (e.g., sharp torso with blurred limbs).

    For fine-grained blur categorisation, use BlurClassifier (CNN) which
    classifies into: sharp, defocused_object_portrait, defocused_blurred,
    motion_blurred.
    """

    def __init__(self, laplacian_threshold: float = 100.0) -> None:
        self.laplacian_threshold = laplacian_threshold

    def detect(
        self, image: np.ndarray, threshold_override: float | None = None
    ) -> dict:
        """Analyze an image for blur.

        Args:
            image: BGR numpy array from cv2.
            threshold_override: If provided, uses this threshold instead of
                the instance default. Thread-safe (no shared state mutation).

        Returns:
            Dict with is_blurry, confidence, and metrics.
        """
        threshold = threshold_override if threshold_override is not None else self.laplacian_threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        if _HAS_CPP:
            laplacian_var = float(_cpp_laplacian_var(gray))
            hf_ratio = float(_cpp_hf_ratio(gray))
        else:
            # Laplacian variance: low variance = blurry
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # FFT-based: ratio of high-frequency energy to total energy
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            cy, cx = h // 2, w // 2
            r = min(h, w) // 8
            mask = np.ones_like(magnitude)
            mask[cy - r : cy + r, cx - r : cx + r] = 0
            total = np.sum(magnitude)
            hf_ratio = float(np.sum(magnitude * mask) / total) if total > 0 else 0.0

        # Normalize Laplacian variance to 640px reference resolution
        # using linear resolution ratio (not pixel count) because Laplacian
        # variance scales with linear resolution, not total pixel count.
        ref_dim = 640
        actual_dim = max(h, w)
        if actual_dim > 0:
            laplacian_var = laplacian_var * ref_dim / actual_dim

        is_blurry = laplacian_var < threshold
        if is_blurry:
            confidence = 1.0 - min(1.0, laplacian_var / threshold)
        else:
            confidence = min(1.0, laplacian_var / threshold)

        return {
            "is_blurry": is_blurry,
            "confidence": confidence,
            "laplacian_variance": laplacian_var,
            "hf_ratio": hf_ratio,
        }
