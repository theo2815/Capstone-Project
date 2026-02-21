"""Placeholder for model download script.

In production, this would download models from S3/GCS with checksum verification.
For development, InsightFace and PaddleOCR download models automatically on first run.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def download_models(manifest_path: str = "./models/manifest.json") -> None:
    setup_logging("INFO")
    manifest = Path(manifest_path)
    if not manifest.exists():
        logger.error("Model manifest not found", path=manifest_path)
        return

    with open(manifest) as f:
        data = json.load(f)

    for model_key, model_info in data.get("models", {}).items():
        model_file = Path("./models") / model_info["file"]
        if model_file.exists() or model_file.is_dir():
            logger.info("Model already present", model=model_key, path=str(model_file))
        else:
            logger.info(
                "Model not found locally",
                model=model_key,
                notes=model_info.get("notes", ""),
            )


if __name__ == "__main__":
    download_models()
