from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np

# Eagerly import the modules that task functions import *lazily* so that
# ``unittest.mock.patch`` can resolve them as ``@patch`` decoration targets.
import src.workers.helpers  # noqa: F401
import src.workers.model_loader  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers -- shorthand constants
# ---------------------------------------------------------------------------
JOB_ID = str(uuid.uuid4())
# Tasks take blob-store paths, not base64 — image bytes never go on the Celery
# queue (see CLAUDE.md "Batch Job Flow"). Workers decode from these paths.
FAKE_PATH = "/tmp/quickpitik-blobs/job/00000.bin"

# A tiny 2x2 BGR image that satisfies `image.shape[:2]`
FAKE_IMAGE = np.zeros((2, 2, 3), dtype=np.uint8)

# A tiny 2x2 grayscale image — the blur detect path decodes grayscale only.
FAKE_GRAY = np.zeros((2, 2), dtype=np.uint8)

# Patch targets -- the task functions do *deferred* imports from these modules,
# so we patch at the source rather than the consuming module.
_MODEL = "src.workers.model_loader"
_HELP = "src.workers.helpers"


# =========================================================================
# blur_detect_batch
# =========================================================================
class TestBlurDetectBatch:
    """Tests for src.workers.tasks.blur_tasks.blur_detect_batch."""

    def _import_task(self):
        from src.workers.tasks.blur_tasks import blur_detect_batch
        return blur_detect_batch

    # ----- normal operation -------------------------------------------------
    @patch(f"{_MODEL}.get_blur_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_grays_from_paths")
    def test_returns_results_for_each_image(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_detector
    ):
        mock_detector = MagicMock()
        mock_detector.detect_fast.return_value = {"is_blurry": True, "score": 0.9}
        mock_get_detector.return_value = mock_detector
        mock_decode.return_value = [FAKE_GRAY, FAKE_GRAY]

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH, FAKE_PATH])

        assert mock_detector.detect_fast.call_count == 2
        mock_complete.assert_called_once()
        results = mock_complete.call_args[0][1]
        assert len(results) == 2
        assert results[0]["index"] == 0
        assert results[0]["is_blurry"] is True
        assert results[1]["index"] == 1
        mock_fail.assert_not_called()

    # ----- model not loaded -------------------------------------------------
    @patch(f"{_MODEL}.get_blur_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_grays_from_paths")
    def test_calls_fail_job_when_detector_is_none(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_detector
    ):
        mock_get_detector.return_value = None

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_fail.assert_called_once_with(JOB_ID, "Blur detector model not loaded in worker")
        mock_complete.assert_not_called()

    # ----- bad image data ---------------------------------------------------
    @patch(f"{_MODEL}.get_blur_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_grays_from_paths")
    def test_records_error_when_decode_returns_none(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_detector
    ):
        mock_get_detector.return_value = MagicMock()
        mock_decode.return_value = [None]  # simulate bad image data

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_complete.assert_called_once()
        results = mock_complete.call_args[0][1]
        assert len(results) == 1
        assert results[0]["error"] == "Failed to decode image"

    # ----- detector raises exception ----------------------------------------
    @patch(f"{_MODEL}.get_blur_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_grays_from_paths")
    def test_records_error_when_detector_raises(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_detector
    ):
        mock_detector = MagicMock()
        mock_detector.detect_fast.side_effect = RuntimeError("inference failure")
        mock_get_detector.return_value = mock_detector
        mock_decode.return_value = [FAKE_GRAY]

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        results = mock_complete.call_args[0][1]
        assert results[0]["error"] == "inference failure"


# =========================================================================
# blur_classify_batch
# =========================================================================
class TestBlurClassifyBatch:
    """Tests for src.workers.tasks.blur_tasks.blur_classify_batch."""

    def _import_task(self):
        from src.workers.tasks.blur_tasks import blur_classify_batch
        return blur_classify_batch

    # ----- full classification (no blur_type) --------------------------------
    @patch(f"{_MODEL}.get_blur_classifier")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_classify_mode_returns_classification(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_classifier
    ):
        mock_classifier = MagicMock()
        mock_classifier.classify_batch.return_value = [
            {"predicted_class": "motion_blurred", "confidence": 0.85, "probabilities": {}}
        ]
        mock_get_classifier.return_value = mock_classifier
        mock_decode.return_value = [FAKE_IMAGE]

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_classifier.classify_batch.assert_called_once_with([FAKE_IMAGE])
        results = mock_complete.call_args[0][1]
        assert results[0]["predicted_class"] == "motion_blurred"
        assert results[0]["confidence"] == 0.85

    # ----- targeted blur_type mode ------------------------------------------
    @patch(f"{_MODEL}.get_blur_classifier")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_blur_type_mode_derives_detected_from_batch(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_classifier
    ):
        """blur_type mode reuses the batched classify_batch result and derives
        detected/blur_type_probability from it, rather than calling
        detect_blur_type per image (which would forfeit the batch)."""
        mock_classifier = MagicMock()
        mock_classifier.min_detection_confidence = 0.5
        mock_classifier.classify_batch.return_value = [{
            "predicted_class": "motion_blurred",
            "confidence": 0.9,
            "probabilities": {"motion_blurred": 0.9, "sharp": 0.1},
        }]
        mock_get_classifier.return_value = mock_classifier
        mock_decode.return_value = [FAKE_IMAGE]

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH], blur_type="motion_blurred")

        mock_classifier.classify_batch.assert_called_once()
        mock_classifier.detect_blur_type.assert_not_called()
        results = mock_complete.call_args[0][1]
        assert results[0]["detected"] is True
        assert results[0]["blur_type"] == "motion_blurred"
        assert results[0]["blur_type_probability"] == 0.9

    @patch(f"{_MODEL}.get_blur_classifier")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_blur_type_mode_applies_min_confidence_floor(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_classifier
    ):
        """A below-floor prediction is NOT detected, matching the sync paths.

        BlurClassifier.detect_blur_type and /blur/classify/stream both gate
        `detected` on min_detection_confidence. This path skipped it, so the
        same photo came back detected=True from /classify/batch and
        detected=False from /classify — a contract split by endpoint.
        """
        mock_classifier = MagicMock()
        mock_classifier.min_detection_confidence = 0.5
        mock_classifier.classify_batch.return_value = [{
            "predicted_class": "motion_blurred",
            "confidence": 0.35,  # matches blur_type, but below the floor
            "probabilities": {"motion_blurred": 0.35, "sharp": 0.33},
        }]
        mock_get_classifier.return_value = mock_classifier
        mock_decode.return_value = [FAKE_IMAGE]

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH], blur_type="motion_blurred")

        results = mock_complete.call_args[0][1]
        assert results[0]["detected"] is False
        assert results[0]["predicted_class"] == "motion_blurred"

    # ----- classifier is None -----------------------------------------------
    @patch(f"{_MODEL}.get_blur_classifier")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_calls_fail_job_when_classifier_is_none(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_classifier
    ):
        mock_get_classifier.return_value = None

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_fail.assert_called_once_with(
            JOB_ID, "Blur classifier model not loaded in worker"
        )
        mock_complete.assert_not_called()

    # ----- classify returns None ---------------------------------------------
    @patch(f"{_MODEL}.get_blur_classifier")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_records_error_when_classify_returns_none(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_classifier
    ):
        mock_classifier = MagicMock()
        mock_classifier.classify_batch.return_value = [None]
        mock_get_classifier.return_value = mock_classifier
        mock_decode.return_value = [FAKE_IMAGE]

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        results = mock_complete.call_args[0][1]
        assert results[0]["error"] == "Classifier returned None"

    # ----- sub-batch raises -> per-image fallback ----------------------------
    @patch(f"{_MODEL}.get_blur_classifier")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_falls_back_to_per_image_when_batch_raises(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_classifier
    ):
        mock_classifier = MagicMock()
        mock_classifier.classify_batch.side_effect = RuntimeError("onnx blew up")
        mock_classifier.classify.return_value = {
            "predicted_class": "sharp", "confidence": 0.99, "probabilities": {},
        }
        mock_get_classifier.return_value = mock_classifier
        mock_decode.return_value = [FAKE_IMAGE]

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_classifier.classify.assert_called_once_with(FAKE_IMAGE)
        results = mock_complete.call_args[0][1]
        assert results[0]["predicted_class"] == "sharp"
        mock_fail.assert_not_called()


# =========================================================================
# face_process_batch
# =========================================================================
class TestFaceProcessBatch:
    """Tests for src.workers.tasks.face_tasks.face_process_batch."""

    def _import_task(self):
        from src.workers.tasks.face_tasks import face_process_batch
        return face_process_batch

    # ----- detect operation -------------------------------------------------
    @patch(f"{_MODEL}.get_face_embedder")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    def test_detect_operation_returns_faces(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_embedder
    ):
        fake_faces = [{"bbox": [10, 20, 50, 60]}, {"bbox": [100, 200, 150, 260]}]
        mock_embedder = MagicMock()
        mock_embedder.detect_faces.return_value = fake_faces
        mock_get_embedder.return_value = mock_embedder
        mock_decode.return_value = FAKE_IMAGE

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH], "detect")

        results = mock_complete.call_args[0][1]
        assert results[0]["faces_detected"] == 2
        assert results[0]["faces"] == fake_faces

    # ----- search operation -------------------------------------------------
    @patch("src.workers.tasks.face_tasks._search_batch")
    @patch(f"{_MODEL}.get_face_embedder")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    def test_search_operation_delegates_to_search_batch(
        self, mock_decode, mock_progress, mock_fail, mock_complete,
        mock_get_embedder, mock_search_batch
    ):
        """search delegates to _search_batch, which fills the shared results
        list in place using ONE DB session + one LATERAL query for all faces."""
        mock_embedder = MagicMock()
        mock_get_embedder.return_value = mock_embedder
        mock_decode.return_value = FAKE_IMAGE

        def fill_results(image_paths, embedder, results, job_id, total, **kwargs):
            results[0] = {
                "index": 0,
                "faces_detected": 1,
                "matches": [{"person_id": "abc", "similarity": 0.95}],
            }

        mock_search_batch.side_effect = fill_results

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH], "search")

        mock_search_batch.assert_called_once()
        results = mock_complete.call_args[0][1]
        assert results[0]["faces_detected"] == 1
        assert results[0]["matches"][0]["person_id"] == "abc"

    # ----- non-search operation takes the detect path -------------------------
    @patch(f"{_MODEL}.get_face_embedder")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    def test_non_search_operation_takes_detect_path(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_embedder
    ):
        """The worker branches on `operation == "search"` only; anything else
        runs detect. Unknown values cannot reach it over HTTP — the route
        constrains operation with pattern="^(detect|search)$"."""
        mock_embedder = MagicMock()
        mock_embedder.detect_faces.return_value = []
        mock_get_embedder.return_value = mock_embedder
        mock_decode.return_value = FAKE_IMAGE

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH], "foobar")

        mock_embedder.detect_faces.assert_called_once()
        results = mock_complete.call_args[0][1]
        assert results[0]["faces_detected"] == 0

    # ----- embedder not loaded -----------------------------------------------
    @patch(f"{_MODEL}.get_face_embedder")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    def test_calls_fail_job_when_embedder_is_none(
        self, mock_decode, mock_progress, mock_fail, mock_complete, mock_get_embedder
    ):
        mock_get_embedder.return_value = None

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH], "detect")

        mock_fail.assert_called_once_with(
            JOB_ID, "Face embedder model not loaded in worker"
        )
        mock_complete.assert_not_called()


# =========================================================================
# bib_recognize_batch
# =========================================================================
class TestBibRecognizeBatch:
    """Tests for src.workers.tasks.bib_tasks.bib_recognize_batch."""

    def _import_task(self):
        from src.workers.tasks.bib_tasks import bib_recognize_batch
        return bib_recognize_batch

    # ----- detector + OCR pipeline ------------------------------------------
    @patch(f"{_MODEL}.get_bib_recognizer")
    @patch(f"{_MODEL}.get_bib_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_detector_plus_ocr_pipeline(
        self, mock_decode_many, mock_decode_one, mock_progress, mock_fail,
        mock_complete, mock_get_detector, mock_get_ocr
    ):
        # Use a 100x100 image so cropping produces a non-empty slice
        big_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_decode_many.return_value = [big_image]

        mock_detector = MagicMock()
        mock_detector.model = True  # truthy -- detector is available
        mock_detector.detect_batch.return_value = [
            [{"bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
        ]
        mock_get_detector.return_value = mock_detector

        mock_ocr = MagicMock()
        mock_ocr.recognize_batch.return_value = [{
            "bib_number": "1234",
            "confidence": 0.92,
            "all_candidates": ["1234"],
        }]
        mock_get_ocr.return_value = mock_ocr

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_detector.detect_batch.assert_called_once()
        mock_ocr.recognize_batch.assert_called_once()
        results = mock_complete.call_args[0][1]
        assert len(results) == 1
        assert results[0]["bibs"][0]["bib_number"] == "1234"

    # ----- OCR-only fallback (no detector) ----------------------------------
    @patch(f"{_MODEL}.get_bib_recognizer")
    @patch(f"{_MODEL}.get_bib_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_ocr_only_fallback_when_detector_is_none(
        self, mock_decode_many, mock_decode_one, mock_progress, mock_fail,
        mock_complete, mock_get_detector, mock_get_ocr
    ):
        big_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_decode_many.return_value = [big_image]
        mock_decode_one.return_value = big_image

        mock_get_detector.return_value = None  # no detector available

        mock_ocr = MagicMock()
        mock_ocr.recognize.return_value = {
            "bib_number": "5678",
            "confidence": 0.88,
            "all_candidates": ["5678"],
        }
        mock_get_ocr.return_value = mock_ocr

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_ocr.recognize.assert_called_once()
        results = mock_complete.call_args[0][1]
        bibs = results[0]["bibs"]
        assert len(bibs) == 1
        assert bibs[0]["bib_number"] == "5678"
        # Fallback should use full-image bbox and be flagged as such
        assert bibs[0]["bbox"] == {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}
        assert bibs[0]["warning"] == "full_image_fallback"

    # ----- OCR-only fallback with model attribute None ----------------------
    @patch(f"{_MODEL}.get_bib_recognizer")
    @patch(f"{_MODEL}.get_bib_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_ocr_fallback_when_detector_model_is_none(
        self, mock_decode_many, mock_decode_one, mock_progress, mock_fail,
        mock_complete, mock_get_detector, mock_get_ocr
    ):
        big_image = np.zeros((80, 120, 3), dtype=np.uint8)
        mock_decode_many.return_value = [big_image]
        mock_decode_one.return_value = big_image

        mock_detector = MagicMock()
        mock_detector.model = None  # detector object exists but its model failed to load
        mock_get_detector.return_value = mock_detector

        mock_ocr = MagicMock()
        mock_ocr.recognize.return_value = {
            "bib_number": "42",
            "confidence": 0.75,
            "all_candidates": ["42"],
        }
        mock_get_ocr.return_value = mock_ocr

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        results = mock_complete.call_args[0][1]
        bibs = results[0]["bibs"]
        assert bibs[0]["bib_number"] == "42"
        # w=120, h=80
        assert bibs[0]["bbox"] == {"x1": 0.0, "y1": 0.0, "x2": 120.0, "y2": 80.0}

    # ----- OCR model not loaded at all --------------------------------------
    @patch(f"{_MODEL}.get_bib_recognizer")
    @patch(f"{_MODEL}.get_bib_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_calls_fail_job_when_ocr_is_none(
        self, mock_decode_many, mock_decode_one, mock_progress, mock_fail,
        mock_complete, mock_get_detector, mock_get_ocr
    ):
        mock_get_ocr.return_value = None

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        mock_fail.assert_called_once_with(JOB_ID, "Bib OCR model not loaded in worker")
        mock_complete.assert_not_called()

    # ----- no bib found (OCR returns empty) ---------------------------------
    @patch(f"{_MODEL}.get_bib_recognizer")
    @patch(f"{_MODEL}.get_bib_detector")
    @patch(f"{_HELP}.complete_job")
    @patch(f"{_HELP}.fail_job")
    @patch(f"{_HELP}.update_job_progress")
    @patch(f"{_HELP}.decode_image_from_path")
    @patch(f"{_HELP}.decode_images_from_paths")
    def test_empty_bibs_when_ocr_finds_nothing(
        self, mock_decode_many, mock_decode_one, mock_progress, mock_fail,
        mock_complete, mock_get_detector, mock_get_ocr
    ):
        big_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_decode_many.return_value = [big_image]
        mock_decode_one.return_value = big_image
        mock_get_detector.return_value = None

        mock_ocr = MagicMock()
        mock_ocr.recognize.return_value = {
            "bib_number": None,
            "confidence": 0.0,
            "all_candidates": [],
        }
        mock_get_ocr.return_value = mock_ocr

        task_fn = self._import_task()
        task_fn(JOB_ID, [FAKE_PATH])

        results = mock_complete.call_args[0][1]
        assert results[0]["bibs"] == []


# =========================================================================
# update_job_progress throttle logic
# =========================================================================
class TestUpdateJobProgressThrottle:
    """Tests for the throttle behaviour in src.workers.helpers.update_job_progress."""

    @patch(f"{_HELP}.get_sync_session")
    def test_skips_write_when_not_on_boundary(self, mock_session):
        from src.workers.helpers import update_job_progress

        # processed=3, total=20, every_n=10  ->  3 < 20 and 3 % 10 != 0  ->  skip
        update_job_progress(JOB_ID, 3, 20, every_n=10)
        mock_session.assert_not_called()

    @patch(f"{_HELP}.get_sync_session")
    def test_writes_on_every_n_boundary(self, mock_session):
        from src.workers.helpers import update_job_progress

        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # processed=10, total=20, every_n=10  ->  10 % 10 == 0  ->  write
        update_job_progress(JOB_ID, 10, 20, every_n=10)
        mock_session.assert_called_once()

    @patch(f"{_HELP}.get_sync_session")
    def test_always_writes_on_final_item(self, mock_session):
        from src.workers.helpers import update_job_progress

        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # processed == total == 17  ->  processed is NOT < total  ->  write
        update_job_progress(JOB_ID, 17, 17, every_n=10)
        mock_session.assert_called_once()

    @patch(f"{_HELP}.get_sync_session")
    def test_skips_several_then_writes_on_boundary(self, mock_session):
        from src.workers.helpers import update_job_progress

        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        total = 25
        for processed in range(1, total + 1):
            update_job_progress(JOB_ID, processed, total, every_n=10)

        # Expected writes: 10, 20, 25 (final) = 3 calls
        assert mock_session.call_count == 3

    @patch(f"{_HELP}.get_sync_session")
    def test_single_item_batch_writes_immediately(self, mock_session):
        from src.workers.helpers import update_job_progress

        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # processed=1, total=1 -> final item -> write
        update_job_progress(JOB_ID, 1, 1, every_n=10)
        mock_session.assert_called_once()
