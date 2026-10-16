from __future__ import annotations

import os

from celery.signals import worker_process_init, worker_process_shutdown

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level globals for ML models
_blur_detector = None
_blur_classifier = None


def get_blur_detector():
    return _blur_detector


def get_blur_classifier():
    return _blur_classifier


def _should_load(queue_name: str) -> bool:
    """Return True if this worker is configured to serve the given queue.

    Reads WORKER_QUEUES env var (comma-separated). Defaults to loading all
    models when the env var is absent (e.g. dev/single-worker deployments).
    """
    worker_queues = os.environ.get("WORKER_QUEUES", "blur,default")
    return queue_name in {q.strip() for q in worker_queues.split(",")}


@worker_process_init.connect
def load_models_on_worker_start(**kwargs):
    """Load only the ML models needed for this worker's queue(s).

    Reads WORKER_QUEUES to determine which models to load:
      blur   → blur detector + classifier
      default → no ML models (webhook delivery, maintenance tasks)

    Unset WORKER_QUEUES (or set to "blur,default") to load everything
    — preserves backward-compat for dev single-worker setups.
    """
    global _blur_detector, _blur_classifier

    from src.config import get_settings
    from src.db.sync_session import init_sync_db

    settings = get_settings()
    worker_queues = os.environ.get("WORKER_QUEUES", "blur,default")

    # Set thread limits before any model loading to prevent over-subscription
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.ONNX_INTRA_OP_THREADS))

    # Initialize sync database engine for this worker process
    init_sync_db()

    logger.info("Worker: starting model load", queues=worker_queues)

    # --- Blur models (only for blur queue workers) ---
    if _should_load("blur"):
        try:
            from src.ml.blur.detector import BlurDetector

            _blur_detector = BlurDetector(laplacian_threshold=settings.BLUR_THRESHOLD)
            logger.info("Worker: blur detector loaded")
        except Exception as e:
            logger.error("Worker: failed to load blur detector", error=str(e))

        try:
            from src.ml.blur.classifier import BlurClassifier

            _blur_classifier = BlurClassifier(
                detector_path=f"{settings.MODEL_DIR}/person_detector/yolov8n_seg_person.onnx",
                rule_path=f"{settings.MODEL_DIR}/blur_classifier/people_rule.json",
                use_gpu=settings.USE_GPU,
                min_detection_confidence=settings.BLUR_DETECTION_MIN_CONFIDENCE,
            )
            if _blur_classifier.session is not None:
                logger.info("Worker: blur classifier loaded")
            else:
                _blur_classifier = None
                logger.warning("Worker: blur classifier model not found, skipping")
        except Exception as e:
            logger.warning("Worker: blur classifier unavailable", error=str(e))

    logger.info("Worker: model loading complete", queues=worker_queues)


@worker_process_shutdown.connect
def cleanup_on_worker_shutdown(**kwargs):
    """Clean up resources when a worker process shuts down."""
    from src.db.sync_session import close_sync_db

    close_sync_db()
    logger.info("Worker: cleanup complete")
