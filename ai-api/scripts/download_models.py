"""Verify that the model artifacts named in the manifest are actually present.

Nothing is downloaded. InsightFace and PaddleOCR fetch their own weights on
first use; the blur classifier and bib detector are produced by the training
scripts in this directory and are mounted into containers from the host
(./models:/app/models:ro in docker-compose.prod.yml) rather than baked into the
image. So the useful job here is the pre-deploy check: say plainly which
artifacts are missing and exit non-zero, because a missing OPTIONAL model is
otherwise invisible — the registry marks blur_classifier and bib_detector
optional, so the API starts and /health/ready returns 200 without them while
every /blur/classify* route fails.

Entries with "file": null (bib_ocr — PaddleOCR caches outside models/) have no
artifact to check and are skipped.

Usage:
    python scripts/download_models.py            # exit 0 = all present
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def download_models(manifest_path: str = "./models/manifest.json") -> int:
    """Report on every manifest entry. Returns the intended process exit code."""
    setup_logging("INFO")
    manifest = Path(manifest_path)
    if not manifest.exists():
        logger.error("Model manifest not found", path=manifest_path)
        return 1

    with open(manifest) as f:
        data = json.load(f)

    models_root = manifest.parent
    missing: list[str] = []

    for model_key, model_info in data.get("models", {}).items():
        rel = model_info.get("file")
        if rel is None:
            logger.info(
                "No local artifact by design, skipping",
                model=model_key,
                notes=model_info.get("notes", ""),
            )
            continue

        model_file = models_root / rel
        if model_file.exists():
            logger.info("Model present", model=model_key, path=str(model_file))
        else:
            missing.append(model_key)
            logger.error(
                "Model MISSING",
                model=model_key,
                path=str(model_file),
                notes=model_info.get("notes", ""),
            )

    if missing:
        logger.error(
            "Model check failed — mount or restore these before deploying",
            missing=",".join(missing),
        )
        return 1

    logger.info("All manifest models present")
    return 0


if __name__ == "__main__":
    sys.exit(download_models())
