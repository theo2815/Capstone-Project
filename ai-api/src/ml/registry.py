from __future__ import annotations

import asyncio
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Singleton model registry. Loads ML models once at startup, serves for app lifetime."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._profile = "all"

    async def load_all(self, settings: Any) -> None:
        """Load all ML models during FastAPI lifespan startup."""
        logger.info("Loading ML models...")
        self._profile = settings.MODEL_PROFILE

        loaders = {
            "blur": self._load_blur,
            "blur_classifier": self._load_blur_classifier,
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
                logger.info("Model load finished", model=name, available=self.is_loaded(name))

    def get(self, name: str) -> Any:
        """Get a loaded model by name. Returns None if not loaded."""
        return self._models.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded and available."""
        model = self._models.get(name)
        return model is not None and (
            name != "blur_classifier" or getattr(model, "session", None) is not None
        )

    def all_loaded(self) -> bool:
        """Check if all required models are loaded.

        Blur classifier is required in the dedicated blur profile.
        """
        required = {"blur", "blur_classifier"} if self._profile == "blur" else {"blur"}
        return all(self.is_loaded(name) for name in required)

    def status(self) -> dict[str, bool]:
        """Per-model usable state for the authenticated readiness payload."""
        return {name: self.is_loaded(name) for name in self._models}

    async def unload_all(self) -> None:
        """Release model memory and GPU/ONNX resources during graceful shutdown."""
        logger.info("Unloading ML models...")
        for name, model in self._models.items():
            if model is None:
                continue
            try:
                # BlurClassifier — ONNX session
                if hasattr(model, "session") and model.session is not None:
                    del model.session
                    model.session = None
                logger.info("Model resources released", model=name)
            except Exception as e:
                logger.warning("Error releasing model resources", model=name, error=str(e))
        self._models.clear()
        logger.info("All models unloaded")

    def _load_blur(self, settings: Any) -> Any:
        from src.ml.blur.detector import BlurDetector

        return BlurDetector(laplacian_threshold=settings.BLUR_THRESHOLD)

    def _load_blur_classifier(self, settings: Any) -> Any:
        from src.ml.blur.classifier import BlurClassifier

        return BlurClassifier(
            detector_path=f"{settings.MODEL_DIR}/person_detector/yolov8n_seg_person.onnx",
            rule_path=f"{settings.MODEL_DIR}/blur_classifier/people_rule.json",
            use_gpu=settings.USE_GPU,
            min_detection_confidence=settings.BLUR_DETECTION_MIN_CONFIDENCE,
            manifest_path=(
                f"{settings.MODEL_DIR}/manifest.json" if settings.MODEL_PROFILE == "blur" else None
            ),
        )
