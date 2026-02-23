from __future__ import annotations

from celery.signals import worker_process_init, worker_process_shutdown

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level globals for ML models
_blur_detector = None
_blur_classifier = None
_face_embedder = None
_bib_detector = None
_bib_recognizer = None


def get_blur_detector():
    return _blur_detector


def get_blur_classifier():
    return _blur_classifier


def get_face_embedder():
    return _face_embedder


def get_bib_detector():
    return _bib_detector


def get_bib_recognizer():
    return _bib_recognizer


@worker_process_init.connect
def load_models_on_worker_start(**kwargs):
    """Load all ML models when a Celery worker process starts."""
    global _blur_detector, _blur_classifier, _face_embedder, _bib_detector, _bib_recognizer

    from src.config import get_settings
    from src.db.sync_session import init_sync_db

    settings = get_settings()

    # Initialize sync database engine for this worker process
    init_sync_db()

    # Load blur detector
    try:
        from src.ml.blur.detector import BlurDetector

        _blur_detector = BlurDetector(laplacian_threshold=settings.BLUR_THRESHOLD)
        logger.info("Worker: blur detector loaded")
    except Exception as e:
        logger.error("Worker: failed to load blur detector", error=str(e))

    # Load blur classifier (optional — requires trained model)
    try:
        from src.ml.blur.classifier import BlurClassifier

        _blur_classifier = BlurClassifier(
            model_path=f"{settings.MODEL_DIR}/blur_classifier/blur_classifier.onnx",
            class_names_path=f"{settings.MODEL_DIR}/blur_classifier/class_names.json",
        )
        if _blur_classifier.session is not None:
            logger.info("Worker: blur classifier loaded")
        else:
            _blur_classifier = None
            logger.warning("Worker: blur classifier model not found, skipping")
    except Exception as e:
        logger.warning("Worker: blur classifier unavailable", error=str(e))

    # Load face embedder
    try:
        from src.ml.faces.embedder import FaceEmbedder

        _face_embedder = FaceEmbedder(
            model_dir=settings.MODEL_DIR,
            use_gpu=settings.USE_GPU,
            det_size=settings.FACE_DET_SIZE,
        )
        logger.info("Worker: face embedder loaded")
    except Exception as e:
        logger.error("Worker: failed to load face embedder", error=str(e))

    # Load bib detector (optional — requires custom YOLO model)
    try:
        from src.ml.bibs.detector import BibDetector

        _bib_detector = BibDetector(
            model_path=f"{settings.MODEL_DIR}/bib_detection/yolov8n_bib.onnx"
        )
        logger.info("Worker: bib detector loaded")
    except Exception as e:
        logger.warning("Worker: bib detector unavailable", error=str(e))

    # Load bib recognizer (PaddleOCR)
    try:
        from src.ml.bibs.recognizer import BibRecognizer

        _bib_recognizer = BibRecognizer(use_gpu=settings.USE_GPU)
        logger.info("Worker: bib recognizer loaded")
    except Exception as e:
        logger.error("Worker: failed to load bib recognizer", error=str(e))

    logger.info("Worker: all model loading complete")


@worker_process_shutdown.connect
def cleanup_on_worker_shutdown(**kwargs):
    """Clean up resources when a worker process shuts down."""
    from src.db.sync_session import close_sync_db

    close_sync_db()
    logger.info("Worker: cleanup complete")
