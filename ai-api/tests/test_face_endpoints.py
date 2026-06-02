"""Integration tests for face recognition endpoints."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def app():
    from src.main import create_app

    return create_app()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _make_jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a JPEG image in memory."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestFaceDetect:
    def test_detect_returns_200(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "faces_detected" in body["data"]
        assert "faces" in body["data"]
        assert "processing_time_ms" in body["data"]

    def test_detect_response_format(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert "success" in body
        assert "request_id" in body
        assert "timestamp" in body

    def test_detect_no_faces_in_random_noise(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("noise.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert body["data"]["faces_detected"] == 0
        assert body["data"]["faces"] == []

    def test_detect_invalid_file_type(self, client: TestClient):
        resp = client.post(
            "/api/v1/faces/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400


class TestFaceEnroll:
    def test_enroll_no_face(self, client: TestClient):
        """Enrolling an image with no face should return NO_FACES error."""
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/enroll",
            files={"file": ("noise.jpg", data, "image/jpeg")},
            data={"person_name": "Test Person"},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NO_FACES"

    def test_enroll_missing_person_name(self, client: TestClient):
        """Enrolling without person_name should fail validation."""
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/enroll",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 422


class TestFaceSearch:
    def test_search_no_face_returns_empty(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search",
            params={"event_id": "11111111-1111-1111-1111-111111111111"},
            files={"file": ("noise.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["faces_detected"] == 0
        assert body["data"]["matches"] == []

    def test_search_response_format(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search",
            params={"event_id": "11111111-1111-1111-1111-111111111111"},
            files={"file": ("noise.jpg", data, "image/jpeg")},
        )
        body = resp.json()
        assert "faces_detected" in body["data"]
        assert "matches" in body["data"]
        assert "unmatched_faces" in body["data"]
        assert "processing_time_ms" in body["data"]


class TestSearchEventIsolation:
    """Fail-closed event isolation: a face search MUST be event-scoped or it
    would match across every event for the key (root rule 5). Omitting event_id
    is rejected with 422 before any DB lookup, instead of silently spanning
    events. Guards the cross-event search regression on every search surface."""

    def test_single_search_requires_event_id(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search",
            files={"file": ("noise.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_batch_search_requires_event_id(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search/batch",
            params={"operation": "search"},
            files=[("files", ("noise.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "EVENT_ID_REQUIRED"

    def test_mega_search_requires_event_id(self, client: TestClient):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search/mega",
            params={"operation": "search"},
            files=[("files", ("noise.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "EVENT_ID_REQUIRED"

    def test_batch_detect_does_not_require_event_id(self, client: TestClient):
        # detect does no DB lookup, so it stays usable without an event scope.
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/search/batch",
            params={"operation": "detect"},
            files=[("files", ("noise.jpg", data, "image/jpeg"))],
        )
        assert resp.status_code != 422


class TestFaceCompare:
    def test_compare_no_faces(self, client: TestClient):
        data1 = _make_jpeg_bytes()
        data2 = _make_jpeg_bytes()
        resp = client.post(
            "/api/v1/faces/compare",
            files=[
                ("file1", ("a.jpg", data1, "image/jpeg")),
                ("file2", ("b.jpg", data2, "image/jpeg")),
            ],
        )
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NO_FACES"


class TestPersonCRUD:
    def test_get_nonexistent_person(self, client: TestClient):
        resp = client.get("/api/v1/faces/persons/00000000-0000-0000-0000-000000000000")
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NOT_FOUND"

    def test_delete_nonexistent_person(self, client: TestClient):
        resp = client.delete("/api/v1/faces/persons/00000000-0000-0000-0000-000000000000")
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "NOT_FOUND"

    def test_invalid_uuid_returns_422(self, client: TestClient):
        resp = client.get("/api/v1/faces/persons/not-a-uuid")
        assert resp.status_code == 422


class TestEnrollMultiFace:
    """Regression guard for the crowd-photo face-indexing fix (2026-06-02).

    A single source image can contain several runners. The
    (person_id, source_image_hash) unique index used to collapse every face
    from one image to a single stored embedding (because the hash was computed
    over the whole image), so only one runner was ever findable in a crowd
    shot. The enroll loop now suffixes the face index, so N faces store N
    embeddings with distinct hashes. Mocks the embedder + DB session because
    the suite has no real multi-face fixture and no test database.
    """

    def test_enroll_stores_distinct_hash_per_face(self, app, client, monkeypatch):
        import contextlib
        import types
        import uuid as _uuid

        captured_hashes: list[str] = []

        # Three high-confidence faces, all from the same uploaded image.
        fake_faces = [
            {"bbox": {"confidence": 0.95}, "embedding": [0.01 * (i + 1)] * 512}
            for i in range(3)
        ]
        fake_embedder = types.SimpleNamespace(get_embeddings=lambda image: fake_faces)

        registry = app.state.model_registry
        orig_get = registry.get
        monkeypatch.setattr(
            registry,
            "get",
            lambda name: fake_embedder if name == "face" else orig_get(name),
        )

        class _FakeRepo:
            def __init__(self, session):
                pass

            async def get_person(self, pid, api_key_id=None):
                return types.SimpleNamespace(id=pid, name="x", event_id=None)

            async def create_person(self, name, api_key_id, event_id):
                return types.SimpleNamespace(id=_uuid.uuid4(), name=name, event_id=event_id)

            async def store_embedding(self, person_id, embedding, source_image_hash, quality_score=None):
                captured_hashes.append(source_image_hash)
                return object()  # truthy => stored

        @contextlib.asynccontextmanager
        async def _fake_session_ctx():
            yield None

        monkeypatch.setattr("src.db.repositories.face_repo.FaceRepository", _FakeRepo)
        monkeypatch.setattr("src.db.session.get_session_ctx", _fake_session_ctx)

        resp = client.post(
            "/api/v1/faces/enroll",
            files={"file": ("crowd.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"person_name": "Event Photo", "event_id": str(_uuid.uuid4())},
        )
        body = resp.json()
        assert body["success"] is True, body
        assert body["data"]["faces_enrolled"] == 3
        # The crux: three faces from ONE image -> three DISTINCT stored hashes.
        assert len(captured_hashes) == 3
        assert len(set(captured_hashes)) == 3
        # Each value is the per-face key "{image_hash}:{i}" — a 64-char sha256
        # hexdigest plus a ":<index>" suffix, so it is LONGER than 64 chars. This
        # is exactly why source_image_hash must be wider than VARCHAR(64) (see
        # migration i9d0e1f2g3h4); the original column would have truncated here.
        for h in captured_hashes:
            base, sep, idx = h.rpartition(":")
            assert sep == ":" and len(base) == 64 and idx.isdigit()
            assert len(h) > 64


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


class TestEnrollMegaBatch:
    """The bulk photo-indexing primitive: ONE person per image, all faces.

    ``face_enroll_mega_batch`` enrolls each uploaded photo as its own ai
    person (unlike ``face_enroll_batch`` which lumps all images under one
    shared person), storing every detected face under it with a per-face
    distinct hash. Runs the task body synchronously with ``.apply()`` and a
    throwaway ``job_id=""`` so it returns the result list directly instead of
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

        from src.workers.tasks.face_tasks import face_enroll_mega_batch

        event_id = str(_uuid.uuid4())
        refs = ["photo-A", "photo-B", "photo-C"]
        results = face_enroll_mega_batch.apply(
            args=["", ["p0", "p1", "p2"]],
            kwargs={"api_key_id": "key-1", "event_id": event_id, "refs": refs},
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
