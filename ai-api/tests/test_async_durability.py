"""Durability regressions for the optional async blur deployment."""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, select, update


def test_blob_references_are_opaque_and_bounded(tmp_path, monkeypatch):
    from src.utils import blob_store

    settings = type("Settings", (), {"BLOB_STORE_PATH": str(tmp_path), "MAX_FILE_SIZE": 4,
                                     "AWS_S3_BUCKET_NAME": ""})()
    monkeypatch.setattr(blob_store, "get_settings", lambda: settings)
    job_id = str(uuid.uuid4())
    keys = blob_store.store_batch(job_id, [b"abcd"])

    assert keys == [f"jobs/{job_id}/00000.bin"]
    assert blob_store.load_blob(keys[0]) == b"abcd"
    (tmp_path / keys[0]).write_bytes(b"abcde")
    with pytest.raises(ValueError, match="size"):
        blob_store.load_blob(keys[0])


def test_blob_store_cleans_partial_upload(tmp_path, monkeypatch):
    from src.utils import blob_store

    settings = type("Settings", (), {"BLOB_STORE_PATH": str(tmp_path), "MAX_FILE_SIZE": 4,
                                     "AWS_S3_BUCKET_NAME": ""})()
    monkeypatch.setattr(blob_store, "get_settings", lambda: settings)
    job_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match="size"):
        blob_store.store_batch(job_id, [b"ok", b"abcde"])
    assert not (tmp_path / "jobs" / job_id).exists()


@pytest.mark.asyncio
async def test_admission_is_atomic_for_same_key():
    from src.api.v1.batch_utils import create_batch_job
    from src.db.models import Job
    from src.db.session import get_session_ctx, init_db

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    settings = SimpleNamespace(MAX_ACTIVE_JOBS_PER_KEY=1, MAX_ACTIVE_JOBS_PER_KEY_INTERNAL=1)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)),
                              state=SimpleNamespace(request_id="test"))
    try:
        results = await asyncio.gather(*[
            create_batch_job(request, "blur_detect_batch", 1, key) for _ in range(2)
        ])
        assert sorted(type(r).__name__ for r in results) == ["JSONResponse", "str"]
        assert next(r for r in results if not isinstance(r, str)).status_code == 429
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.api_key_id == key))


@pytest.mark.asyncio
async def test_terminal_result_cannot_be_overwritten():
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job = await JobRepository(session).create("blur_detect_batch", 1, key)
        job_id = job.id
    try:
        with get_sync_session() as session:
            repo = SyncJobRepository(session)
            repo.complete(job_id, [{"index": 0, "is_blurry": False}])
            repo.fail(job_id, "redelivered after cleanup")
            repo.update_progress(job_id, 0, 0.0)
        with get_sync_session() as session:
            job = SyncJobRepository(session).get(job_id)
            assert job.status == "completed"
            assert job.result == [{"index": 0, "is_blurry": False}]
            assert job.progress == 1.0
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == job_id))


@pytest.mark.asyncio
async def test_expired_processing_attempt_can_be_reclaimed_without_stale_commit():
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job = await JobRepository(session).create("blur_detect_batch", 1, key)
        job_id = job.id
    try:
        with get_sync_session() as session:
            repo = SyncJobRepository(session)
            assert repo.claim(job_id, "first") == "claimed"
            assert repo.claim(job_id, "second") == "busy"
        with get_sync_session() as session:
            session.execute(update(Job).where(Job.id == job_id).values(
                lease_expires_at=datetime.now(UTC) - timedelta(seconds=1)))
        with get_sync_session() as session:
            repo = SyncJobRepository(session)
            repo.reap_stale_jobs(4200)
            assert repo.get(job_id).status == "processing"
            assert repo.claim(job_id, "second") == "claimed"
            assert not repo.complete(job_id, [{"index": 0, "value": "old"}], "first")
            assert repo.complete(job_id, [{"index": 0, "value": "new"}], "second")
            assert repo.claim(job_id, "third") == "terminal"
        with get_sync_session() as session:
            job = SyncJobRepository(session).get(job_id)
            assert job.result == [{"index": 0, "value": "new"}]
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == job_id))


@pytest.mark.asyncio
async def test_duplicate_completion_keeps_result_and_inputs_until_retention(tmp_path, monkeypatch):
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session
    from src.utils import blob_store
    from src.workers import helpers

    await init_db()
    settings = SimpleNamespace(BLOB_STORE_PATH=str(tmp_path), AWS_S3_BUCKET_NAME="",
                               MAX_FILE_SIZE=10)
    monkeypatch.setattr(blob_store, "get_settings", lambda: settings)
    events = []
    monkeypatch.setattr(helpers, "dispatch_webhook_sync", lambda **kw: events.append(kw))
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job = await JobRepository(session).create("blur_detect_batch", 1, key)
        job_id = str(job.id)
    try:
        blob_key = blob_store.store_batch(job_id, [b"image"])[0]
        helpers.complete_job(job_id, [{"index": 0, "value": "first"}])
        helpers.complete_job(job_id, [{"index": 0, "value": "second"}])
        assert blob_store.load_blob(blob_key) == b"image"
        with get_sync_session() as session:
            assert SyncJobRepository(session).get(uuid.UUID(job_id)).result == [
                {"index": 0, "value": "first"}]
        assert len(events) == 1
    finally:
        blob_store.cleanup_batch(job_id)
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == uuid.UUID(job_id)))


@pytest.mark.asyncio
async def test_reaper_removes_terminal_inputs_but_keeps_running_job(tmp_path, monkeypatch):
    import os
    import time

    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.session import get_session_ctx, init_db
    from src.utils import blob_store
    from src.workers.tasks import maintenance_tasks

    await init_db()
    settings = SimpleNamespace(BLOB_STORE_PATH=str(tmp_path), AWS_S3_BUCKET_NAME="",
                               MAX_FILE_SIZE=10, BLOB_RETENTION_SECONDS=2)
    monkeypatch.setattr(blob_store, "get_settings", lambda: settings)
    monkeypatch.setattr("src.config.get_settings", lambda: settings)
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        repo = JobRepository(session)
        done = await repo.create("blur_detect_batch", 1, key)
        active = await repo.create("blur_detect_batch", 1, key)
        done_id, active_id = str(done.id), str(active.id)
    try:
        done_key = blob_store.store_batch(done_id, [b"done"])[0]
        active_key = blob_store.store_batch(active_id, [b"live"])[0]
        old = time.time() - 10
        os.utime(tmp_path / "jobs" / done_id, (old, old))
        os.utime(tmp_path / "jobs" / active_id, (old, old))
        async with get_session_ctx() as session:
            await session.execute(update(Job).where(Job.id == uuid.UUID(done_id)).values(
                status="completed", completed_at=datetime.now(UTC) - timedelta(seconds=10)))
            await session.execute(update(Job).where(Job.id == uuid.UUID(active_id)).values(
                status="processing", lease_expires_at=datetime.now(UTC) + timedelta(hours=1)))
        assert maintenance_tasks.cleanup_stale_blobs.run() == 1
        assert not (tmp_path / done_key).exists()
        assert blob_store.load_blob(active_key) == b"live"
    finally:
        blob_store.cleanup_batch(active_id)
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.api_key_id == key))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["upload", "broker"])
async def test_blur_submission_failure_is_terminal_and_cleans_staging(
    failure, tmp_path, monkeypatch,
):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from PIL import Image

    from src.api.v1 import blur
    from src.config import get_settings
    from src.db.models import Job
    from src.db.session import get_session_ctx, init_db
    from src.middleware.auth import verify_api_key
    from src.utils import blob_store
    from src.workers.tasks.blur_tasks import blur_detect_batch

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_ACTIVE_JOBS_PER_KEY", 0)
    monkeypatch.setattr(settings, "BLOB_STORE_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "AWS_S3_BUCKET_NAME", "")
    app = FastAPI()
    app.state.settings = settings
    app.include_router(blur.router, prefix="/api/v1")
    app.dependency_overrides[verify_api_key] = lambda: {
        "key_id": key, "scopes": ["blur:read"], "rate_tier": "pro"}
    def raise_failure(*_):
        raise OSError(failure)

    if failure == "upload":
        monkeypatch.setattr(blob_store, "store_batch", raise_failure)
    else:
        monkeypatch.setattr(blur_detect_batch, "delay", raise_failure)
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buf, format="JPEG")
    try:
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False),
                               base_url="http://test") as client:
            response = await client.post("/api/v1/blur/detect/batch", files={
                "files": ("photo.jpg", buf.getvalue(), "image/jpeg")})
        assert response.status_code == 503
        async with get_session_ctx() as session:
            jobs = (await session.execute(select(Job).where(Job.api_key_id == key))).scalars().all()
            assert len(jobs) == 1
            assert jobs[0].status == "failed"
            assert not (tmp_path / "jobs" / str(jobs[0].id)).exists()
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.api_key_id == key))


@pytest.mark.asyncio
async def test_redelivered_terminal_blur_task_returns_before_reading_input(monkeypatch):
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session
    from src.workers.tasks import blur_tasks

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job = await JobRepository(session).create("blur_detect_batch", 1, key)
        job_id = str(job.id)
    with get_sync_session() as session:
        SyncJobRepository(session).complete(uuid.UUID(job_id), [{"index": 0, "is_blurry": False}])
    from src.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "MODEL_PROFILE", "blur")
    monkeypatch.setattr("src.workers.model_loader.get_blur_detector", lambda: object())
    monkeypatch.setattr("src.workers.helpers.decode_grays_from_paths",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("input read")))
    try:
        result = blur_tasks.blur_detect_batch.run(job_id, ["jobs/missing/00000.bin"])
        assert result == [{"index": 0, "is_blurry": False}]
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == uuid.UUID(job_id)))


@pytest.mark.asyncio
async def test_failed_mega_child_marks_parent_once(monkeypatch):
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session
    from src.workers.tasks import blur_tasks, maintenance_tasks

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job = await JobRepository(session).create("blur_detect_mega", 1, key)
        job_id = str(job.id)
    monkeypatch.setattr("src.workers.model_loader.get_blur_detector", lambda: None)
    try:
        with pytest.raises(RuntimeError, match="model"):
            blur_tasks.blur_detect_batch.run(job_id, ["jobs/missing/00000.bin"], mega_child=True)
        maintenance_tasks.fail_mega_batch.run("failed-child", job_id)
        monkeypatch.setattr("src.workers.helpers.decode_grays_from_paths",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("input read")))
        assert blur_tasks.blur_detect_batch.run(
            job_id, ["jobs/missing/00000.bin"], mega_child=True) is None
        maintenance_tasks.finalize_mega_batch.run([[{"index": 0, "is_blurry": False}]], job_id)
        with get_sync_session() as session:
            job = SyncJobRepository(session).get(uuid.UUID(job_id))
            assert job.status == "failed"
            assert job.result is None
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == uuid.UUID(job_id)))


@pytest.mark.asyncio
async def test_mega_heartbeat_prevents_age_only_reaping():
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job_id = (await JobRepository(session).create("blur_detect_mega", 51, key)).id
        await session.execute(update(Job).where(Job.id == job_id).values(
            created_at=datetime.now(UTC) - timedelta(hours=3)))
    try:
        with get_sync_session() as session:
            repo = SyncJobRepository(session)
            repo.touch_mega(job_id, 3900)
            repo.reap_stale_jobs(4200)
            assert repo.get(job_id).status == "processing"
            assert repo.fail(job_id, "child failed")
            assert not repo.touch_mega(job_id, 3900)
        with get_sync_session() as session:
            assert SyncJobRepository(session).get(job_id).status == "failed"
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == job_id))


@pytest.mark.asyncio
async def test_mega_finalizer_retry_keeps_first_result(monkeypatch):
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session
    from src.workers import helpers
    from src.workers.tasks import maintenance_tasks

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job_id = (await JobRepository(session).create("blur_detect_mega", 2, key)).id
    events = []
    monkeypatch.setattr(helpers, "dispatch_webhook_sync", lambda **kw: events.append(kw))
    try:
        maintenance_tasks.finalize_mega_batch.run(
            [[{"index": 0, "value": "first"}], [{"index": 0, "value": "second"}]], str(job_id))
        maintenance_tasks.finalize_mega_batch.run(
            [[{"index": 0, "value": "changed"}], [{"index": 0, "value": "changed"}]], str(job_id))
        with get_sync_session() as session:
            job = SyncJobRepository(session).get(job_id)
            assert job.result == [{"index": 0, "value": "first"},
                                  {"index": 1, "value": "second"}]
            assert len(events) == 1
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == job_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["input", "model"])
async def test_missing_async_dependency_has_terminal_result(missing, monkeypatch):
    from src.config import get_settings
    from src.db.models import Job
    from src.db.repositories.job_repo import JobRepository
    from src.db.repositories.sync_job_repo import SyncJobRepository
    from src.db.session import get_session_ctx, init_db
    from src.db.sync_session import get_sync_session
    from src.workers.tasks import blur_tasks

    await init_db()
    key = f"durability-test-{uuid.uuid4()}"
    async with get_session_ctx() as session:
        job_id = (await JobRepository(session).create("blur_detect_batch", 1, key)).id
    monkeypatch.setattr(get_settings(), "MODEL_PROFILE", "blur")
    monkeypatch.setattr("src.workers.model_loader.get_blur_detector",
                        lambda: None if missing == "model" else object())
    try:
        blur_tasks.blur_detect_batch.run(str(job_id), [f"jobs/{job_id}/00000.bin"])
        with get_sync_session() as session:
            job = SyncJobRepository(session).get(job_id)
            if missing == "model":
                assert job.status == "failed"
            else:
                assert job.status == "completed"
                assert job.result == [{"index": 0, "error": "Failed to decode image"}]
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.id == job_id))


@pytest.mark.skipif(not os.getenv("TASK10B_S3_ENDPOINT"), reason="local S3 test store unavailable")
def test_s3_batch_is_readable_by_separate_process_and_private(monkeypatch):
    from src.utils import blob_store

    settings = SimpleNamespace(
        AWS_S3_BUCKET_NAME=os.environ["TASK10B_S3_BUCKET"],
        AWS_ENDPOINT_URL=os.environ["TASK10B_S3_ENDPOINT"],
        AWS_DEFAULT_REGION="us-east-1", AWS_S3_URL_STYLE="path", MAX_FILE_SIZE=10,
    )
    monkeypatch.setattr(blob_store, "get_settings", lambda: settings)
    job_id = str(uuid.uuid4())
    key = blob_store.store_batch(job_id, [b"photo"])[0]
    try:
        result = subprocess.run([sys.executable, "-c",
            "from src.utils.blob_store import load_blob; import sys; "
            "assert load_blob(sys.argv[1]) == b'photo'", key],
            env={**os.environ, "AWS_S3_BUCKET_NAME": settings.AWS_S3_BUCKET_NAME,
                 "AWS_ENDPOINT_URL": settings.AWS_ENDPOINT_URL,
                 "AWS_DEFAULT_REGION": "us-east-1", "AWS_S3_URL_STYLE": "path"},
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        denied = subprocess.run([sys.executable, "-c",
            "from botocore.exceptions import ClientError\n"
            "from src.utils.blob_store import load_blob\n"
            "import sys\n"
            "try:\n    load_blob(sys.argv[1])\n"
            "except ClientError as exc:\n"
            "    assert exc.response['Error']['Code'] in ('InvalidAccessKeyId', 'AccessDenied')\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n", key],
            env={**os.environ, "AWS_S3_BUCKET_NAME": settings.AWS_S3_BUCKET_NAME,
                 "AWS_ENDPOINT_URL": settings.AWS_ENDPOINT_URL,
                 "AWS_DEFAULT_REGION": "us-east-1", "AWS_S3_URL_STYLE": "path",
                 "AWS_ACCESS_KEY_ID": "unrelated", "AWS_SECRET_ACCESS_KEY": "unrelated"},
            capture_output=True, text=True)
        assert denied.returncode == 0, denied.stderr
    finally:
        blob_store.cleanup_batch(job_id)


@pytest.mark.skipif(not os.getenv("TASK10B_API_URL"), reason="separate API/worker not running")
@pytest.mark.asyncio
@pytest.mark.parametrize("route,count", [("batch", 1), ("mega", 51)])
async def test_separate_api_and_worker_complete_private_async_job(route, count):
    import secrets
    import time

    from httpx import AsyncClient
    from PIL import Image

    from src.db.models import APIKey, Job
    from src.db.session import get_session_ctx, init_db

    await init_db()
    raw_keys = [secrets.token_urlsafe(32) for _ in range(2)]
    ids = [uuid.uuid4() for _ in range(2)]
    async with get_session_ctx() as session:
        for ident, raw in zip(ids, raw_keys, strict=True):
            session.add(APIKey(
                id=ident, name="task10b-local-test",
                key_hash=hashlib.sha256(raw.encode()).hexdigest(),
                scopes=["blur:read", "jobs:read"], rate_tier="internal", active=True,
            ))
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buf, format="JPEG")
    job_id = None
    try:
        async with AsyncClient(base_url=os.environ["TASK10B_API_URL"], timeout=20) as client:
            response = await client.post(f"/api/v1/blur/detect/{route}",
                headers={"X-API-Key": raw_keys[0]},
                files=[("files", (f"photo-{i}.jpg", buf.getvalue(), "image/jpeg"))
                       for i in range(count)])
            assert response.status_code == 202, response.text
            job_id = response.json()["data"]["job_id"]
            deadline = time.monotonic() + 45
            while True:
                poll = await client.get(f"/api/v1/jobs/{job_id}",
                                        headers={"X-API-Key": raw_keys[0]})
                assert poll.status_code == 200
                data = poll.json()["data"]
                if data["status"] != "pending" and data["status"] != "processing":
                    break
                assert time.monotonic() < deadline, data
                await asyncio.sleep(0.25)
            assert data["status"] == "completed", data
            assert data["result_total"] == count
            assert [row["index"] for row in data["result"]] == list(range(min(count, 100)))
            assert "is_blurry" in data["result"][0]
            other = await client.get(f"/api/v1/jobs/{job_id}",
                                     headers={"X-API-Key": raw_keys[1]})
            assert other.json()["error"]["code"] == "NOT_FOUND"
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.api_key_id.in_([str(i) for i in ids])))
            await session.execute(delete(APIKey).where(APIKey.id.in_(ids)))


@pytest.mark.skipif(not os.getenv("TASK10B_BAD_WORKER"), reason="bad-model worker not running")
@pytest.mark.asyncio
async def test_separate_worker_failed_mega_child_marks_parent():
    import secrets
    import time

    from httpx import AsyncClient
    from PIL import Image

    from src.db.models import APIKey, Job
    from src.db.session import get_session_ctx, init_db

    await init_db()
    raw = secrets.token_urlsafe(32)
    ident = uuid.uuid4()
    async with get_session_ctx() as session:
        session.add(APIKey(id=ident, name="task10b-bad-worker-test",
                           key_hash=hashlib.sha256(raw.encode()).hexdigest(),
                           scopes=["blur:read", "jobs:read"], rate_tier="internal", active=True))
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buf, format="JPEG")
    try:
        async with AsyncClient(base_url=os.environ["TASK10B_API_URL"], timeout=20) as client:
            response = await client.post("/api/v1/blur/classify/mega",
                headers={"X-API-Key": raw},
                files={"files": ("photo.jpg", buf.getvalue(), "image/jpeg")})
            assert response.status_code == 202, response.text
            job_id = response.json()["data"]["job_id"]
            deadline = time.monotonic() + 45
            while True:
                poll = await client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": raw})
                assert poll.status_code == 200
                data = poll.json()["data"]
                if data["status"] not in ("pending", "processing"):
                    break
                assert time.monotonic() < deadline, data
                await asyncio.sleep(0.25)
            assert data["status"] == "failed", data
            assert data["result"] is None
    finally:
        async with get_session_ctx() as session:
            await session.execute(delete(Job).where(Job.api_key_id == str(ident)))
            await session.execute(delete(APIKey).where(APIKey.id == ident))
