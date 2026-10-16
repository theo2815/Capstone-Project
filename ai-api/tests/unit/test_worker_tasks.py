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


class TestEnrollMegaBatch:
    """The bulk photo-indexing primitive: ONE person per image, all faces.

    ``face_enroll_mega_batch`` enrolls each uploaded photo as its own ai
    person (unlike ``face_enroll_batch`` which lumps all images under one
    shared person), storing every detected face under it with a per-face
    distinct hash. Runs the task body synchronously with ``.apply()`` and a
    valid parent ID with ``mega_child=True`` so it returns the result list instead of
    writing to the DB / firing webhooks. Mocks the embedder, blob loader, and
    sync session because the suite has no test database.
    """

    def test_one_person_per_image_all_faces_with_ref(self, monkeypatch):
        import contextlib
        import types
        import uuid as _uuid

        # image 0: a 3-runner crowd; image 1: a single runner; image 2: no face.
        face_sets = [
            [{"bbox": {"confidence": 0.95}, "embedding": [0.01 * (i + 1)] * 512} for i in range(3)],
            [{"bbox": {"confidence": 0.9}, "embedding": [0.5] * 512}],
            [],
        ]
        calls = {"n": 0}

        def _embed(image):
            r = face_sets[calls["n"]]
            calls["n"] += 1
            return r

        fake_embedder = types.SimpleNamespace(get_embeddings=_embed)

        monkeypatch.setattr("src.workers.model_loader.get_face_embedder", lambda: fake_embedder)
        # Distinct bytes per path => distinct per-image sha256 base hash.
        monkeypatch.setattr("src.utils.blob_store.load_blob", lambda path: b"raw-" + path.encode())
        monkeypatch.setattr("src.workers.helpers._decode_raw_bytes", lambda raw, max_dim=0: object())

        created_persons: list[tuple] = []
        stored: list[tuple] = []

        class _Savepoint:
            def commit(self):
                pass

            def rollback(self):
                pass

        class _FakeSession:
            def begin_nested(self):
                return _Savepoint()

        @contextlib.contextmanager
        def _fake_sync_session():
            yield _FakeSession()

        class _FakeRepo:
            def __init__(self, session):
                pass

            def create_person(self, name, api_key_id=None, event_id=None):
                pid = _uuid.uuid4()
                created_persons.append((pid, name, event_id))
                return types.SimpleNamespace(id=pid, name=name, event_id=event_id)

            def bulk_store_embeddings(self, person_id, faces, min_conf=0.0):
                hashes = [f["source_image_hash"] for f in faces]
                stored.append((person_id, hashes))
                return len(faces), 0

        monkeypatch.setattr("src.db.sync_session.get_sync_session", _fake_sync_session)
        monkeypatch.setattr("src.db.repositories.sync_face_repo.SyncFaceRepository", _FakeRepo)
        monkeypatch.setattr("src.workers.helpers.touch_mega_job", lambda _: True)

        from src.workers.tasks.face_tasks import face_enroll_mega_batch

        event_id = str(_uuid.uuid4())
        refs = ["photo-A", "photo-B", "photo-C"]
        results = face_enroll_mega_batch.apply(
            args=[str(_uuid.uuid4()), ["p0", "p1", "p2"]],
            kwargs={"api_key_id": "key-1", "event_id": event_id,
                    "refs": refs, "mega_child": True},
        ).get()

        assert len(results) == 3
        # image 0 — crowd: one person, all 3 faces, ref echoed
        assert results[0]["ref"] == "photo-A"
        assert results[0]["person_id"] is not None
        assert results[0]["faces_enrolled"] == 3
        # image 1 — single runner
        assert results[1]["ref"] == "photo-B"
        assert results[1]["faces_enrolled"] == 1
        # image 2 — no enrollable face: benign, no person created
        assert results[2]["ref"] == "photo-C"
        assert results[2]["person_id"] is None
        assert results[2]["faces_enrolled"] == 0
        # exactly TWO persons (images 0 + 1) — not one shared, not three
        assert len(created_persons) == 2
        # event isolation: every person stamped with the SAME event_id
        assert all(ev == event_id for (_, _, ev) in created_persons)
        # crowd photo stored three DISTINCT per-face hashes
        crowd_hashes = stored[0][1]
        assert len(crowd_hashes) == 3 and len(set(crowd_hashes)) == 3


class TestVectorCastBinding:
    """Vector casts in raw SQL must use CAST(:name AS vector), not :name::vector.

    SQLAlchemy's text() bind-parameter regex refuses a name followed by ':',
    so ":query::vector" backtracks and binds a parameter called "quer". The
    real value is never substituted, Postgres receives a literal ':' and
    raises `syntax error at or near ":"`.

    This was fixed in the async FaceRepository on 2026-05-20 but the sync copy
    kept the broken form until 2026-08-14 — which meant /faces/search/batch and
    /faces/search/mega failed for every image containing exactly ONE face (the
    len(embeddings) == 1 branch of batch_search_similar delegates to
    search_similar). The worker swallows per-image exceptions into a result
    row, so it degraded silently rather than failing the job.

    DB-free: asserts on how the SQL text parses, not on query results.
    """

    def _binds(self, sql: str) -> set[str]:
        from sqlalchemy import text

        return set(text(sql)._bindparams.keys())

    def test_colon_cast_loses_the_parameter(self):
        # Documents WHY the rule exists — this is the shape that broke.
        assert self._binds("SELECT :query::vector") == {"quer"}
        assert self._binds("SELECT CAST(:query AS vector)") == {"query"}

    @pytest.mark.parametrize(
        "module",
        [
            "src.db.repositories.sync_face_repo",
        ],
    )
    def test_repos_contain_no_colon_cast(self, module: str):
        import ast
        import importlib
        import inspect
        import re

        source = inspect.getsource(importlib.import_module(module))

        # Drop docstrings before scanning. The SQL we are policing lives in
        # ordinary string literals, so the scan cannot simply skip strings —
        # but docstrings are prose, and the ones explaining this very bug
        # quote the broken form on purpose.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    source = source.replace(doc, "")

        offenders = re.findall(r":\w+::\w+", source)
        assert not offenders, f"{module} uses ':name::type' casts: {offenders}"


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


class TestWorkerDecodeDimensions:
    """The face and bib workers must decode at the same dimension as the
    single-image endpoints.

    Until 2026-08-14 they hardcoded their own: face 768, bib 1024, while
    /faces/* and /bibs/recognize decoded at MAX_INFERENCE_DIMENSION (1280).
    Same photo, different result depending on which path indexed it — a photo
    enrolled by the batch drain got a different embedding than the same photo
    enrolled synchronously, and 17 of 80 measured race photos read a DIFFERENT
    bib string at 1024 than at 1280. Bib numbers are exact-match lookup keys,
    so that divergence mis-attributes photos to runners.

    Blur is deliberately exempt and is asserted separately below: its two
    calibrated dims are shared with their own endpoints, which is the property
    that matters. Source-level assertions — no models, no DB.
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

    def test_face_worker_uses_the_global_dimension(self):
        calls = self._decode_calls("src.workers.tasks.face_tasks")
        assert calls, "no decode calls found — test is looking at the wrong module"
        overrides = [c for c in calls if "<default>" not in c]
        assert not overrides, f"face worker overrides the decode dim: {overrides}"

    def test_bib_worker_uses_the_global_dimension(self):
        calls = self._decode_calls("src.workers.tasks.bib_tasks")
        assert calls, "no decode calls found — test is looking at the wrong module"
        overrides = [c for c in calls if "<default>" not in c]
        assert not overrides, f"bib worker overrides the decode dim: {overrides}"

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
