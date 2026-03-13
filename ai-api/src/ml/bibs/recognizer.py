from __future__ import annotations

import os
import re

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

_BIB_CHAR_RE = re.compile(r"[A-Za-z0-9\-_]")

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

    def recognize(self, cropped_bib_image: np.ndarray) -> dict:
        """Run OCR on a cropped bib region image.

        Args:
            cropped_bib_image: BGR numpy array of the cropped bib region.

        Returns:
            Dict with bib_number, confidence, and all_candidates.
        """
        # PaddleOCR 3.x uses predict() — ocr() is deprecated
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
            digit_count = sum(c.isdigit() for c in cleaned)
            if digit_count >= self.min_chars:
                candidates.append({"text": cleaned, "confidence": float(score)})

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        best = candidates[0] if candidates else {"text": "", "confidence": 0.0}

        return {
            "bib_number": best["text"],
            "confidence": best["confidence"],
            "all_candidates": candidates,
        }
