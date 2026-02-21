from __future__ import annotations

import asyncio
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Singleton model registry. Loads ML models once at startup, serves for app lifetime."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    async def load_all(self, settings: Any) -> None:
        """Load all ML models during FastAPI lifespan startup."""
        logger.info("Loading ML models...")

        loaders = {
            "blur": self._load_blur,
            "face": self._load_face,
            "bib_detector": self._load_bib_detector,
            "bib_ocr": self._load_bib_ocr,
        }

        results = await asyncio.gather(
            *[asyncio.to_thread(loader, settings) for loader in loaders.values()],
            return_exceptions=True,
        )

        for name, result in zip(loaders.keys(), results):
            if isinstance(result, Exception):
                logger.error("Failed to load model", model=name, error=str(result))
                self._models[name] = None
            else:
                self._models[name] = result
                logger.info("Model loaded", model=name)

    def get(self, name: str) -> Any:
        """Get a loaded model by name. Returns None if not loaded."""
        return self._models.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded and available."""
        return self._models.get(name) is not None

    def all_loaded(self) -> bool:
        """Check if all required models are loaded.

        bib_detector is optional (requires custom-trained YOLO model).
        """
        required = {"blur", "face", "bib_ocr"}
        return all(self.is_loaded(name) for name in required)

    async def unload_all(self) -> None:
        """Release model memory during graceful shutdown."""
        logger.info("Unloading ML models...")
        self._models.clear()
        logger.info("All models unloaded")

    def _load_blur(self, settings: Any) -> Any:
        from src.ml.blur.detector import BlurDetector

        return BlurDetector(laplacian_threshold=settings.BLUR_THRESHOLD)

    def _load_face(self, settings: Any) -> Any:
        from src.ml.faces.embedder import FaceEmbedder

        return FaceEmbedder(
            model_dir=settings.MODEL_DIR,
            use_gpu=settings.USE_GPU,
            det_size=settings.FACE_DET_SIZE,
        )

    def _load_bib_detector(self, settings: Any) -> Any:
        from src.ml.bibs.detector import BibDetector

        return BibDetector(
            model_path=f"{settings.MODEL_DIR}/bib_detection/yolov8n_bib.onnx",
        )

    def _load_bib_ocr(self, settings: Any) -> Any:
        from src.ml.bibs.recognizer import BibRecognizer

        return BibRecognizer(use_gpu=settings.USE_GPU)
