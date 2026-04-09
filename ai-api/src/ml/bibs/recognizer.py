from __future__ import annotations

import os
import re

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

_BIB_CHAR_RE = re.compile(r"[A-Za-z0-9\-_]")

# Common OCR character confusions for bib numbers (mostly numeric)
_OCR_SUBSTITUTIONS = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
})

# PaddleOCR 3.x does a slow connectivity check on import; bypass it
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Disable oneDNN by default — works around a PaddlePaddle 3.x PIR+oneDNN bug on Windows
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")


class BibRecognizer:
    """OCR for bib number text recognition using PaddleOCR 3.x (PP-OCRv5)."""

    def __init__(self, use_gpu: bool = False, min_chars: int = 2) -> None:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x removed use_gpu; device is auto-detected or set via env
        if not use_gpu:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        self.min_chars = min_chars
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
        )
        logger.info("BibRecognizer initialized", gpu=use_gpu)

    def recognize_batch(
        self,
        crops: list[np.ndarray],
        min_chars_override: int | None = None,
    ) -> list[dict]:
        """Run OCR on multiple cropped bib images using a thread pool.

        PaddleOCR 3.x does not expose true GPU-batched prediction via its Python
        API (each predict() call is processed individually internally). We use a
        ThreadPoolExecutor to run multiple predict() calls concurrently, which
        still yields a meaningful speedup because PaddleOCR releases the GIL
        during its I/O and C++ inference portions.

        Args:
            crops: List of BGR numpy arrays (cropped bib regions).
            min_chars_override: Override minimum digit count for every call.

        Returns:
            List of result dicts in the same order as ``crops``.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not crops:
            return []

        results: list[dict | None] = [None] * len(crops)

        from src.config import get_settings

        settings = get_settings()
        max_workers = min(settings.OCR_MAX_WORKERS, len(crops))
        per_crop_timeout = settings.INFERENCE_TIMEOUT
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.recognize, crop, min_chars_override): i
                for i, crop in enumerate(crops)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result(timeout=per_crop_timeout)
                except TimeoutError:
                    results[idx] = {"bib_number": "", "confidence": 0.0, "all_candidates": []}
                    logger.warning("OCR timed out for crop", index=idx)
                except Exception as e:
                    results[idx] = {"bib_number": "", "confidence": 0.0, "all_candidates": []}
                    logger.error("OCR failed in batch", index=idx, error=str(e))

        return [r if r is not None else {"bib_number": "", "confidence": 0.0, "all_candidates": []} for r in results]

    def recognize(
        self, cropped_bib_image: np.ndarray, min_chars_override: int | None = None
    ) -> dict:
        """Run OCR on a cropped bib region image.

        Args:
            cropped_bib_image: BGR numpy array of the cropped bib region.
            min_chars_override: Override minimum digit count for this call (thread-safe).

        Returns:
            Dict with bib_number, confidence, and all_candidates.
        """
        min_chars = min_chars_override if min_chars_override is not None else self.min_chars
        # PaddleOCR 3.x uses predict() — ocr() is deprecated.
        # Called directly (no timeout thread) — the ThreadPoolExecutor in
        # recognize_batch() provides per-crop timeout via future.result(timeout=),
        # and Celery task_soft_time_limit is the outer safety net.
        results = list(self.ocr.predict(cropped_bib_image))
        if not results:
            return {"bib_number": "", "confidence": 0.0, "all_candidates": []}

        # PaddleOCR 3.x returns OCRResult objects (dict-like) with:
        #   rec_texts: list[str], rec_scores: list[float]
        result = results[0]
        rec_texts = result.get("rec_texts", [])
        rec_scores = result.get("rec_scores", [])

        if not rec_texts:
            return {"bib_number": "", "confidence": 0.0, "all_candidates": []}

        candidates = []
        for text, score in zip(rec_texts, rec_scores):
            cleaned = "".join(_BIB_CHAR_RE.findall(str(text))).strip("-_")
            cleaned = cleaned.translate(_OCR_SUBSTITUTIONS)
            digit_count = sum(c.isdigit() for c in cleaned)
            if digit_count >= min_chars:
                candidates.append({"text": cleaned, "confidence": float(score)})

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        best = candidates[0] if candidates else {"text": "", "confidence": 0.0}

        return {
            "bib_number": best["text"],
            "confidence": best["confidence"],
            "all_candidates": candidates,
        }
