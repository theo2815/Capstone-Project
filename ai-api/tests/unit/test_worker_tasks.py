from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


@pytest.fixture(autouse=True)
def _no_job_lease(monkeypatch):
    # The blur profile takes a database lease first; these tests need no database
    # (the lease itself is tested in tests/test_async_durability.py).
    monkeypatch.setattr(f"{_HELP}.claim_job", lambda job_id: ("claimed", None, None))


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


class TestWorkerDecodeDimensions:
    """Blur workers must decode at the same calibrated dimension as their own
    endpoints. Source-level assertions — no models, no DB.
    """

    def _decode_calls(self, module_name: str) -> list[str]:
        """Every decode helper call in a worker module, as source text."""
        import ast
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module_name))
        decoders = {
            "decode_image_from_path", "decode_images_from_paths",
            "_decode_raw_bytes", "decode_gray_from_path", "decode_grays_from_paths",
        }
        calls = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in decoders:
                    kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
                    calls.append(f"{node.func.id}({kw.get('max_dim', '<default>')})")
        return calls

    def test_blur_classify_worker_shares_the_endpoint_setting(self):
        # Blur classify legitimately differs from MAX_INFERENCE_DIMENSION, but
        # the worker and /blur/classify/stream must read the SAME setting.
        calls = self._decode_calls("src.workers.tasks.blur_tasks")
        assert any("BLUR_CLASSIFY_DECODE_DIM" in c for c in calls), calls

    def test_default_global_dimension_is_the_calibrated_1280(self):
        from src.config import Settings

        assert Settings().MAX_INFERENCE_DIMENSION == 1280


def test_worker_starts_the_blur_classifier_with_the_two_stage_two_files(monkeypatch):
    """The worker's call must fit BlurClassifier's real arguments: a keyword the
    class does not accept is swallowed by the loader and leaves no classifier."""
    import inspect

    import src.ml.blur.classifier as classifier_module
    import src.workers.model_loader as loader

    real_signature = inspect.signature(classifier_module.BlurClassifier)

    class FakeClassifier:
        session = object()

        def __init__(self, **kwargs):
            real_signature.bind(**kwargs)
            self.kwargs = kwargs

    monkeypatch.setenv("WORKER_QUEUES", "blur")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr("src.db.sync_session.init_sync_db", lambda: None)
    monkeypatch.setattr(classifier_module, "BlurClassifier", FakeClassifier)
    monkeypatch.setattr(loader, "_blur_detector", None)
    monkeypatch.setattr(loader, "_blur_classifier", None)

    loader.load_models_on_worker_start()

    kwargs = loader.get_blur_classifier().kwargs
    assert kwargs["detector_path"].endswith("person_detector/yolov8n_seg_person.onnx")
    assert kwargs["rule_path"].endswith("blur_classifier/people_rule.json")


class TestWebhookTenantScoping:
    """Job-completion webhooks must fan out only to the owning tenant.

    ``dispatch_webhook_sync`` now threads the job's ``api_key_id`` into
    ``list_by_event`` so a tenant's job.completed never dispatches to another
    tenant's subscriptions (was: event-name-only lookup → cross-tenant leak).
    """

    def test_dispatch_scopes_lookup_by_api_key_id(self, monkeypatch):
        import contextlib

        captured: dict = {}

        class _FakeWebhookRepo:
            def __init__(self, session):
                pass

            def list_by_event(self, event, api_key_id=None):
                captured["event"] = event
                captured["api_key_id"] = api_key_id
                return []  # no subscriptions => no delivery dispatched

        @contextlib.contextmanager
        def _fake_sync_session():
            yield object()

        # dispatch_webhook_sync binds these names at the helpers module top.
        monkeypatch.setattr("src.workers.helpers.get_sync_session", _fake_sync_session)
        monkeypatch.setattr("src.workers.helpers.SyncWebhookRepository", _FakeWebhookRepo)

        from src.workers.helpers import dispatch_webhook_sync

        count = dispatch_webhook_sync(
            "job.completed", {"job_id": "j1", "result_count": 5}, api_key_id="key-A"
        )
        assert count == 0
        assert captured["event"] == "job.completed"
        assert captured["api_key_id"] == "key-A"


class TestSourceImageHashColumn:
    """The face_embeddings.source_image_hash column must fit the per-face key
    "{sha256_hexdigest}:{face_index}" (66+ chars). A VARCHAR(64) — the original
    width — raises StringDataRightTruncation on every enroll against a real
    database. Guards migration i9d0e1f2g3h4 from being reverted or the model
    reset to String(64). DB-free: inspects the ORM column definition only.
    """

    def test_column_fits_per_face_key(self):
        import hashlib

        from src.db.models import FaceEmbedding

        col_len = FaceEmbedding.__table__.c.source_image_hash.type.length
        sample = f"{hashlib.sha256(b'x').hexdigest()}:{999}"
        assert col_len is not None and col_len >= len(sample), (
            f"source_image_hash VARCHAR({col_len}) too narrow for the "
            f"{len(sample)}-char per-face key"
        )
