"""Guards on the production deployment surface.

Every finding these cover shared one root cause: nothing ever ran the real
production artifacts. `nginx.conf` could not parse for months while a test that
regexed one directive out of it passed; the prod image copied a models/
directory that `.dockerignore` had already emptied; the GPU overlay named a
service the prod stack does not define. Reading a value out of a config file
proves very little, so where it is affordable these hand the artifact to the
tool that actually consumes it.

Companion to TestBodySizeLimit in test_blur_endpoint.py, which owns the
nginx.conf <-> MAX_REQUEST_BODY pairing.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shlex
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_forwards_approved_limits_into_smoke_container():
    workflow = yaml.safe_load((PROJECT_ROOT.parent / ".github/workflows/ai-api-blur.yml")
                              .read_text(encoding="utf-8"))
    steps = workflow["jobs"]["release-check"]["steps"]
    smoke = next(step for step in steps if step.get("name", "").startswith("Smoke test metrics"))
    command = next(line for line in smoke["run"].splitlines()
                   if "blur_release_smoke.py" in line)
    args = shlex.split(command)
    forwarded = {args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "-e"}
    assert {"BLUR_MIN_ACCURACY", "BLUR_MAX_MISS_RATE", "BLUR_MAX_OVERCULL_RATE",
            "BLUR_MIN_SHARP", "BLUR_MIN_BLUR", "BLUR_MIN_EVENTS",
            "BLUR_MAX_PHOTOMETRIC_FLIP_RATE"} <= forwarded


PROD_COMPOSE = PROJECT_ROOT / "docker-compose.prod.yml"
GPU_COMPOSE = PROJECT_ROOT / "docker-compose.gpu.yml"

# The API plus every Celery worker. All three mount models and all three decrypt
# webhook secrets, so anything below that says "all app services" means these.
APP_SERVICES = (
    "ai-api",
    "celery-blur",
    "celery-default",
)


@pytest.mark.parametrize("override", [
    {"INFERENCE_SUB_BATCH_SIZE": 0},
    {"BLUR_DECODE_CONCURRENCY": 0},
    {"BLUR_MAX_ACTIVE_REQUESTS": 0},
    {"BLUR_DETECTION_MIN_CONFIDENCE": 1.1},
    {"BLUR_MAX_REQUEST_BODY": 1},
    {"MAX_REQUEST_BODY": 1},
    {"MODEL_PROFILE": "blur", "BLUR_MAX_ACTIVE_REQUESTS": 3},
    {"MODEL_PROFILE": "blur", "BLUR_DECODE_CONCURRENCY": 3},
    {"MODEL_PROFILE": "blur", "INFERENCE_SUB_BATCH_SIZE": 51},
    {"BLUR_INFERENCE_TIMEOUT_SECONDS": 0},
    {"BLUR_STREAM_TIMEOUT_SECONDS": 0},
    {"BLUR_REQUEST_TIMEOUT_SECONDS": 0},
])
def test_invalid_upload_budget_fails_at_config_load(override):
    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)


def test_blur_request_timeout_uses_profile_deadline():
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False,
                        BLUR_REQUEST_TIMEOUT_SECONDS=0.01)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()

    @app.get("/api/v1/blur/slow")
    async def slow():
        await asyncio.sleep(0.08)
        return {"ok": True}

    response = TestClient(app).get("/api/v1/blur/slow")
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "REQUEST_TIMEOUT"


def test_blur_readiness_fails_while_timed_out_native_work_runs(monkeypatch):
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings
    app.state.model_registry = SimpleNamespace(all_loaded=lambda: True, status=lambda: {})
    app.state.redis = SimpleNamespace(ping=lambda: asyncio.sleep(0, result=True))

    async def healthy_db():
        return True

    monkeypatch.setattr("src.db.session.check_db_health", healthy_db)
    monkeypatch.setattr("src.utils.timeout.inference_healthy", lambda: False, raising=False)
    assert TestClient(app).get("/api/v1/health/readyz").status_code == 503


def test_blur_profile_defaults_to_one_process_and_two_cpu_threads():
    from src.config import Settings

    blur = Settings(_env_file=None, MODEL_PROFILE="blur")
    full = Settings(_env_file=None, MODEL_PROFILE="all")
    assert (blur.WORKERS, blur.ONNX_INTRA_OP_THREADS, blur.ONNX_INTER_OP_THREADS) == (1, 2, 1)
    assert (full.WORKERS, full.ONNX_INTRA_OP_THREADS, full.ONNX_INTER_OP_THREADS) == (2, 6, 4)


def test_timed_out_request_keeps_admission_until_native_work_exits():
    from src.api.v1.blur import BlurAuthRoute
    from src.config import Settings
    from src.main import create_app
    from src.utils.timeout import run_with_timeout, start_timeout_executor

    start_timeout_executor()
    # 0.2 s, not less: the first request spends ~30 ms in middleware before the route
    # runs, and native work still queued at the deadline is cancelled by design, so a
    # shorter deadline makes `entered` race and fail.
    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False,
                        BLUR_MAX_ACTIVE_REQUESTS=1, BLUR_REQUEST_TIMEOUT_SECONDS=0.2,
                        BLUR_QUEUE_WAIT_SECONDS=0)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()

    entered = threading.Event()
    release = threading.Event()

    def native():
        entered.set()
        release.wait(5)

    blur = APIRouter(prefix="/api/v1/blur", route_class=BlurAuthRoute)

    @blur.get("/slow-native")
    async def slow_native():
        await asyncio.to_thread(run_with_timeout, native, timeout_seconds=5)
        return {"ok": True}

    @blur.get("/probe")
    async def probe():
        return {"ok": True}

    app.include_router(blur)

    async def valid_key(request, api_key):
        return {"key_id": api_key or "k1", "scopes": ["blur:read"], "rate_tier": "pro"}

    @asynccontextmanager
    async def no_lifespan(_app):
        yield

    app.router.lifespan_context = no_lifespan
    with (patch("src.api.v1.blur.verify_api_key", valid_key),
          TestClient(app, raise_server_exceptions=False) as client):
        try:
            started = time.monotonic()
            assert client.get("/api/v1/blur/slow-native").status_code == 504
            assert time.monotonic() - started < 0.5
            assert entered.wait(1)
            blocked = client.get("/api/v1/blur/probe", headers={"X-API-Key": "other"})
            assert blocked.status_code == 503
            assert blocked.json()["error"]["code"] == "BLUR_BUSY"
        finally:
            release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if client.get("/api/v1/blur/probe").status_code == 200:
                break
            time.sleep(0.01)
        else:
            pytest.fail("admission slot was not released after native work exited")


async def test_bad_key_request_does_not_take_a_blur_slot():
    from src.api.v1.blur import BlurAuthRoute
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False, BLUR_MAX_ACTIVE_REQUESTS=1)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    probe = APIRouter(prefix="/api/v1/blur", route_class=BlurAuthRoute)

    @probe.get("/probe")
    async def probe_route():
        return {"ok": True}

    app.include_router(probe)
    entered, gate = asyncio.Event(), asyncio.Event()

    async def fake_verify(request, api_key):
        if api_key == "bad":
            entered.set()
            await gate.wait()  # a slow bad-key lookup, as under a flood
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {"key_id": "k1", "scopes": ["blur:read"], "rate_tier": "pro"}

    with patch("src.api.v1.blur.verify_api_key", fake_verify):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            bad = asyncio.create_task(
                client.get("/api/v1/blur/probe", headers={"X-API-Key": "bad"})
            )
            await asyncio.wait_for(entered.wait(), 2)
            good = await client.get("/api/v1/blur/probe", headers={"X-API-Key": "good"})
            gate.set()
            assert (await bad).status_code == 401
    assert good.status_code == 200, good.text


async def test_blur_slot_cannot_be_taken_after_the_request_ended():
    # A cancelled route can still resume one step (anyio skips a task whose awaited
    # result is ready), so its key check may reach admit after the middleware cleaned up.
    from src.api.v1.blur import BlurAuthRoute
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False, BLUR_MAX_ACTIVE_REQUESTS=1)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    admits = []

    @app.get("/api/v1/blur/capture")
    async def capture(request: Request):
        admits.append(request.state.blur_admit)
        return {"ok": True}

    probe = APIRouter(prefix="/api/v1/blur", route_class=BlurAuthRoute)

    @probe.get("/probe")
    async def probe_route():
        return {"ok": True}

    app.include_router(probe)

    async def valid_key(request, api_key):
        return {"key_id": "k1", "scopes": ["blur:read"], "rate_tier": "pro"}

    with patch("src.api.v1.blur.verify_api_key", valid_key):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/v1/blur/capture")).status_code == 200
            assert await admits[0]("k1") == "busy"
            probed = await client.get("/api/v1/blur/probe")
    assert probed.status_code == 200, probed.text


async def test_slow_stream_upload_has_a_request_deadline():
    import httpx

    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False, DEBUG=True,
                        ENVIRONMENT="test", BLUR_REQUEST_TIMEOUT_SECONDS=0.02)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings
    app.state.model_registry = SimpleNamespace(get=lambda _name: None)

    async def slow_body():
        yield (
            b'--x\r\nContent-Disposition: form-data; name="files"; filename="a.jpg"\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
        )
        await asyncio.sleep(0.08)
        yield b'not-an-image\r\n--x--\r\n'

    start = time.monotonic()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as client:
        response = await client.post(
            "/api/v1/blur/classify/stream", content=slow_body(),
            headers={"Content-Type": "multipart/form-data; boundary=x"},
        )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "REQUEST_TIMEOUT"
    assert time.monotonic() - start < 0.5


@pytest.mark.asyncio
async def test_stream_deadline_cancels_stalled_response_send():
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False,
                        BLUR_STREAM_TIMEOUT_SECONDS=0.02)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    middleware = next(
        item.cls for item in app.user_middleware
        if item.cls.__name__ == "BlurTimeoutMiddleware"
    )
    cancelled = asyncio.Event()

    async def stream(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        try:
            await send({"type": "http.response.body", "body": b"line\n", "more_body": True})
        finally:
            cancelled.set()

    async def blocked_send(message):
        if message["type"] == "http.response.body":
            await asyncio.Event().wait()

    scope = {"type": "http", "path": "/api/v1/blur/classify/stream", "state": {}}
    await asyncio.wait_for(
        middleware(stream)(scope, lambda: None, blocked_send), timeout=1.2
    )
    assert cancelled.is_set()


def test_blur_profile_stream_reports_incomplete_before_transport_deadline():
    import json

    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", DEBUG=True, METRICS_ENABLED=False,
                        ENVIRONMENT="test", BLUR_STREAM_TIMEOUT_SECONDS=0.02)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings

    class Classifier:
        min_detection_confidence = 0.5
        _dynamic_batch = True

        def classify_batch(self, images):
            time.sleep(0.08)
            return [{"predicted_class": "sharp", "confidence": 1.0,
                     "probabilities": {"sharp": 1.0}} for _ in images]

    app.state.model_registry = SimpleNamespace(get=lambda name: Classifier())
    image = io.BytesIO()
    Image.new("RGB", (64, 64)).save(image, format="JPEG")
    response = TestClient(app).post(
        "/api/v1/blur/classify/stream",
        files={"files": ("image.jpg", image.getvalue(), "image/jpeg")},
    )
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert lines[-1]["_summary"] and lines[-1]["complete"] is False


def _services(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["services"]


def _env(service: dict) -> dict[str, str]:
    """Compose `environment` as a list of KEY=VALUE strings -> dict."""
    out: dict[str, str] = {}
    for item in service.get("environment", []):
        key, _, value = item.partition("=")
        out[key] = value
    return out


class TestProdComposeModels:
    """The full Compose profile still mounts its complete model directory."""

    @pytest.mark.parametrize("name", APP_SERVICES)
    def test_every_app_service_mounts_models(self, name):
        service = _services(PROD_COMPOSE)[name]
        mounts = [m for m in service.get("volumes", []) if ":/app/models" in m]
        assert mounts, f"{name} does not mount ./models — it will run with no weights"
        assert all(m.endswith(":ro") for m in mounts), (
            f"{name} mounts models writable; read-only is what makes a missing "
            f"bundle fail loudly instead of silently re-downloading ~300 MB"
        )

    def test_dockerfile_copies_the_audited_blur_pair(self):
        """The blur image carries its audited model and labels."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "COPY models/ /app/models/" in dockerfile

    def test_dockerignore_keeps_only_the_blur_pair_and_manifest(self):
        lines = [
            ln.strip()
            for ln in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        assert "models/" in lines
        assert "!models/manifest.json" in lines
        assert "!models/blur_classifier/people_rule.json" in lines
        assert "models/person_detector/*" in lines
        assert "!models/person_detector/yolov8n_seg_person.onnx" in lines
        assert "!models/blur_classifier/blur_classifier.onnx" not in lines
        assert "!models/blur_classifier/class_names.json" not in lines
        # Training-Images/ alone is ~16 GB. Leaving it in the context meant every
        # build shipped it to the daemon before the first instruction ran.
        for heavy in ("Training-Images/", "runs/", "build/"):
            assert heavy in lines, f"{heavy} back in the build context"


class TestProdComposeExposure:
    def test_nginx_is_the_only_public_entry_point(self):
        """nginx enforces the rate limit, the 2 GB body cap and (once enabled)
        TLS. Publishing 0.0.0.0:8000 alongside it made all three optional — and
        is why an nginx.conf that could not parse went unnoticed for months.

        nginx itself is exempt: being reachable on 80/443 is its job.
        """
        for name, service in _services(PROD_COMPOSE).items():
            if name == "nginx":
                continue
            for port in service.get("ports", []):
                assert str(port).startswith("127.0.0.1:"), (
                    f"{name} publishes {port} on all interfaces; bind it to "
                    f"127.0.0.1 and let nginx be the only public entry point"
                )

    def test_no_wildcard_cors_default(self):
        """`${ALLOWED_ORIGINS:-["*"]}` gave a deploy that merely forgot the
        variable wildcard CORS in production, against docs/security.md.
        """
        api_env = _env(_services(PROD_COMPOSE)["ai-api"])
        origins = api_env["ALLOWED_ORIGINS"]
        assert '"*"' not in origins, "wildcard CORS default is back"
        assert origins.startswith("${ALLOWED_ORIGINS:?"), (
            "ALLOWED_ORIGINS must fail closed — an unset value should abort "
            "startup, not silently resolve to a permissive default"
        )

    @pytest.mark.parametrize("name", APP_SERVICES)
    def test_webhook_secret_key_reaches_every_service(self, name):
        """The API encrypts (webhook_repo) and the blur/face/bib workers decrypt
        inside complete_job/fail_job (workers/helpers). A partial rollout is
        worse than none: signatures break rather than degrade.
        """
        assert "WEBHOOK_SECRET_KEY" in _env(_services(PROD_COMPOSE)[name]), (
            f"{name} has no WEBHOOK_SECRET_KEY — it cannot decrypt what the "
            f"API encrypted"
        )


class TestGpuOverlay:
    def test_overlay_only_names_services_the_prod_stack_defines(self):
        """It used to override `celery-worker`, which exists only in the dev
        compose — so against prod it added a stray fifth worker and left the
        four real ones on CPU.
        """
        prod = set(_services(PROD_COMPOSE))
        stray = set(_services(GPU_COMPOSE)) - prod
        assert not stray, f"GPU overlay defines services absent from prod: {sorted(stray)}"


class TestReadinessVisibility:
    """The readiness payload is what an orchestrator routes on, so it belongs to
    the deployment surface: a container with no models mounted answered 200 and
    kept taking traffic.
    """

    @staticmethod
    def _registry(**loaded: bool):
        from src.ml.registry import ModelRegistry

        registry = ModelRegistry()
        registry._models = {
            name: (SimpleNamespace(session=object()) if name == "blur_classifier" else object())
            if ok else None for name, ok in loaded.items()
        }
        return registry

    def test_status_reports_every_model_including_the_optional_ones(self):
        registry = self._registry(blur=True, blur_classifier=False)
        assert registry.status() == {"blur": True, "blur_classifier": False}

    def test_the_gap_status_exists_to_expose(self):
        """Not a redundant assertion — this pins the exact hole. The verdict
        stays green with the optional classifier missing, which is why the payload
        needs the per-model detail alongside it. If someone later makes it
        required, this failing is the intended signal to update the docs and the
        deployment guide, not to delete the test.
        """
        registry = self._registry(blur=True, blur_classifier=False)
        assert registry.all_loaded() is True
        assert registry.status()["blur_classifier"] is False

    def test_a_missing_required_model_still_fails_the_verdict(self):
        registry = self._registry(blur=False, blur_classifier=True)
        assert registry.all_loaded() is False

    def test_classifier_without_a_session_is_unavailable(self):
        from src.ml.registry import ModelRegistry

        registry = ModelRegistry()
        registry._profile = "blur"
        registry._models = {
            "blur": object(),
            "blur_classifier": SimpleNamespace(session=None),
        }
        assert registry.status()["blur_classifier"] is False
        assert registry.all_loaded() is False

    def test_public_readyz_reflects_model_and_dependency_recovery(self, monkeypatch):
        from src.config import Settings
        from src.main import create_app
        from src.ml.registry import ModelRegistry
        from src.utils.timeout import start_timeout_executor

        start_timeout_executor()
        settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False)
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()
        app.state.settings = settings
        registry = ModelRegistry()
        registry._profile = "blur"
        registry._models = {"blur": object(), "blur_classifier": SimpleNamespace(session=None)}
        app.state.model_registry = registry

        class Redis:
            available = False

            async def ping(self):
                if not self.available:
                    raise ConnectionError("redis down")

            async def set(self, *_args, **_kwargs):
                return True

            async def get(self, *_args):
                return "ready"

            async def eval(self, *_args):
                return [1, 0, 1]

        redis = Redis()
        app.state.redis = redis
        db = {"available": False}

        async def check_db():
            return db["available"]

        monkeypatch.setattr("src.db.session.check_db_health", check_db)
        client = TestClient(app)
        path = "/api/v1/health/readyz"
        assert client.get("/api/v1/health").status_code == 200
        assert client.get(path).status_code == 503
        db["available"] = True
        redis.available = True
        assert client.get(path).status_code == 503
        registry._models["blur_classifier"].session = object()
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
        redis.available = False
        assert client.get(path).status_code == 503

    def test_public_readyz_bounds_a_stalled_database_check(self, monkeypatch):
        import asyncio

        from src.config import Settings
        from src.main import create_app
        from src.ml.registry import ModelRegistry

        settings = Settings(
            MODEL_PROFILE="blur", METRICS_ENABLED=False, READINESS_TIMEOUT_SECONDS=0.01
        )
        with patch("src.main.get_settings", return_value=settings):
            app = create_app()
        app.state.settings = settings
        registry = ModelRegistry()
        registry._profile = "blur"
        registry._models = {
            "blur": object(), "blur_classifier": SimpleNamespace(session=object())
        }
        app.state.model_registry = registry
        app.state.redis = None

        async def stuck_db():
            await asyncio.sleep(60)
            return True

        monkeypatch.setattr("src.db.session.check_db_health", stuck_db)
        start = time.monotonic()
        assert TestClient(app).get("/api/v1/health/readyz").status_code == 503
        assert time.monotonic() - start < 0.5

    def test_readiness_schema_carries_the_field(self):
        from src.schemas.common import ReadinessResponse

        payload = ReadinessResponse(
            models_loaded=True,
            models={"blur": True, "blur_classifier": False},
            database=True,
            redis=True,
        ).model_dump()
        assert payload["models"]["blur_classifier"] is False


def test_metrics_enabled_health_request_works_without_lifespan():
    from src.config import Settings
    from src.main import create_app

    settings = Settings(METRICS_ENABLED=True)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings
    assert TestClient(app).get("/api/v1/health").status_code == 200


def test_metrics_requires_explicit_operations_scope():
    import hashlib
    import json

    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", ENVIRONMENT="production", DEBUG=False,
                        METRICS_ENABLED=True)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings

    keys = {
        hashlib.sha256(key.encode()).hexdigest(): json.dumps({
            "key_id": key, "scopes": scopes, "rate_tier": "internal",
        })
        for key, scopes in (
            ("desktop", ["blur:read", "jobs:read"]),
            ("wildcard", ["*"]),
            ("operator", ["metrics:read"]),
        )
    }

    class Redis:
        async def get(self, name):
            return keys.get(name.removeprefix("apikey:"))

        async def eval(self, *_args):
            return [1, 10, 1]

    app.state.redis = Redis()
    client = TestClient(app)
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "desktop"}).status_code == 403
    assert client.get("/metrics", headers={"X-API-Key": "wildcard"}).status_code == 403
    response = client.get("/metrics", headers={"X-API-Key": "operator"})
    assert response.status_code == 200
    assert "http_request_duration_seconds" in response.text


def test_blur_metrics_require_one_api_process():
    from src.config import Settings

    with pytest.raises(ValidationError, match="WORKERS=1"):
        Settings(_env_file=None, MODEL_PROFILE="blur", METRICS_ENABLED=True, WORKERS=2)


def test_native_inference_timeout_is_counted_until_work_exits():
    from prometheus_client import generate_latest

    from src.utils.timeout import InferenceTimeoutError, run_with_timeout, start_timeout_executor

    start_timeout_executor()
    release = threading.Event()

    def count():
        for line in generate_latest().splitlines():
            if line.startswith(b'blur_timeouts_total{phase="native"} '):
                return float(line.rsplit(b" ", 1)[-1])
        return 0.0

    before = count()
    try:
        with pytest.raises(InferenceTimeoutError):
            run_with_timeout(lambda: release.wait(1), timeout_seconds=0.01)
        assert count() - before == 1
    finally:
        release.set()


def test_stream_image_error_is_counted_separately_from_request_rejection():
    import json

    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=True, DEBUG=True, ENVIRONMENT="test")
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings
    app.state.model_registry = SimpleNamespace(
        get=lambda name: SimpleNamespace(min_detection_confidence=0.5)
        if name == "blur_classifier" else None
    )
    client = TestClient(app)

    def metric(name):
        for line in client.get("/metrics").text.splitlines():
            if line.startswith(name + " "):
                return float(line.rsplit(" ", 1)[-1])
        return 0.0

    attempted = 'blur_images_total{operation="classify",outcome="attempted"}'
    failed = 'blur_images_total{operation="classify",outcome="failed"}'
    rejected = 'blur_rejected_requests_total{reason="busy"}'
    before = (metric(attempted), metric(failed), metric(rejected))
    response = client.post("/api/v1/blur/classify/stream",
                           files=[("files", ("bad.jpg", b"not an image", "image/jpeg"))])
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert lines[0]["error"]
    assert lines[-1]["_summary"] and lines[-1]["errors"] == 1
    assert metric(attempted) - before[0] == 1
    assert metric(failed) - before[1] == 1
    assert metric(rejected) - before[2] == 0


def test_readyz_recovers_when_redis_was_down_at_startup(monkeypatch):
    from src.config import Settings
    from src.main import create_app
    from src.ml.registry import ModelRegistry

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False)
    monkeypatch.setattr("src.main.get_settings", lambda: settings)

    async def no_op():
        pass

    async def unload_models(self):
        pass

    async def load_models(self, _settings):
        self._profile = "blur"
        self._models = {
            "blur": object(), "blur_classifier": SimpleNamespace(session=object())
        }

    monkeypatch.setattr("src.main.init_db", no_op)
    monkeypatch.setattr("src.main.close_db", no_op)
    monkeypatch.setattr("src.db.session.check_db_health", lambda: no_op_true())
    monkeypatch.setattr(ModelRegistry, "load_all", load_models)
    monkeypatch.setattr(ModelRegistry, "unload_all", unload_models)

    async def no_op_true():
        return True

    class Redis:
        available = False

        async def ping(self):
            if not self.available:
                from redis.exceptions import ConnectionError as RedisConnectionError

                raise RedisConnectionError("redis unavailable")

        async def set(self, *_args, **_kwargs):
            return True

        async def get(self, key):
            if not self.available:
                from redis.exceptions import ConnectionError as RedisConnectionError

                raise RedisConnectionError("redis unavailable")
            if key == "apikey:readyz":
                return "ready"
            return '{"key_id":"test","scopes":["blur:read"],"rate_tier":"free"}'

        async def eval(self, *_args):
            return [1, 0, 1]

        async def aclose(self):
            pass

    redis = Redis()
    monkeypatch.setattr("redis.asyncio.from_url", lambda *_args, **_kwargs: redis)
    with TestClient(create_app()) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/health/readyz").status_code == 503
        from prometheus_client import generate_latest

        assert b'blur_dependency_ready{dependency="redis"} 0.0' in generate_latest()
        path = "/api/v1/blur/detect"
        headers = {"X-API-Key": "test"}
        assert client.post(path, headers=headers).status_code == 503
        redis.available = True
        assert client.get("/api/v1/health/readyz").status_code == 200
        assert b'blur_dependency_ready{dependency="redis"} 1.0' in generate_latest()
        assert client.post(path, headers=headers).status_code == 422


def test_readyz_requires_redis_auth_and_limiter_commands(monkeypatch):
    from src.config import Settings
    from src.main import create_app
    from src.ml.registry import ModelRegistry
    from src.utils.timeout import start_timeout_executor

    start_timeout_executor()
    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings
    registry = ModelRegistry()
    registry._profile = "blur"
    registry._models = {
        "blur": object(), "blur_classifier": SimpleNamespace(session=object())
    }
    app.state.model_registry = registry

    class Redis:
        fail_at = None

        async def ping(self):
            return True

        async def set(self, *_args, **_kwargs):
            if self.fail_at == "set":
                raise ConnectionError("set unavailable")
            return True

        async def get(self, *_args):
            if self.fail_at == "get":
                raise ConnectionError("get unavailable")
            return "ready"

        async def eval(self, *_args):
            if self.fail_at == "eval":
                raise ConnectionError("eval unavailable")
            return [1, 0, 1]

    redis = Redis()
    app.state.redis = redis

    async def healthy_db():
        return True

    monkeypatch.setattr("src.db.session.check_db_health", healthy_db)
    client = TestClient(app)
    assert client.get("/api/v1/health/readyz").status_code == 200
    for command in ("set", "get", "eval"):
        redis.fail_at = command
        assert client.get("/api/v1/health/readyz").status_code == 503
        assert client.get("/api/v1/health").status_code == 200
    redis.fail_at = None
    assert client.get("/api/v1/health/readyz").status_code == 200


@pytest.mark.asyncio
async def test_blur_profile_loads_only_blur_models(monkeypatch):
    from src.ml.registry import ModelRegistry

    registry = ModelRegistry()
    called = []
    for name in ("blur", "blur_classifier"):
        monkeypatch.setattr(
            registry, f"_load_{name}",
            lambda _settings, n=name: called.append(n) or (
                SimpleNamespace(session=object()) if n == "blur_classifier" else object()
            )
        )
    await registry.load_all(SimpleNamespace(MODEL_PROFILE="blur"))

    assert set(called) == {"blur", "blur_classifier"}
    assert set(registry.status()) == {"blur", "blur_classifier"}
    assert registry.all_loaded()
    registry._models["blur_classifier"] = None
    assert not registry.all_loaded()


def test_blur_profile_omits_face_and_bib_routes():
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=True)
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings
    client = TestClient(app)
    assert client.get("/api/v1/health").status_code == 200
    assert client.post("/api/v1/faces/detect").status_code == 404
    assert client.post("/api/v1/bibs/recognize").status_code == 404


@pytest.mark.parametrize("path", (
    "/api/v1/blur/detect/batch",
    "/api/v1/blur/classify/batch",
    "/api/v1/blur/detect/mega",
    "/api/v1/blur/classify/mega",
))
def test_disabled_blur_async_rejects_before_multipart_or_job_creation(path):
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", BLUR_ASYNC_ENABLED=False,
                        METRICS_ENABLED=False, DEBUG=True, ENVIRONMENT="test")
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings

    with patch("src.api.v1.blur.create_batch_job", side_effect=AssertionError("job created")), \
         patch("src.api.v1.blur.store_blobs_and_get_paths", side_effect=AssertionError("blob written")), \
         patch("src.api.v1.blur.dispatch_mega_batch", side_effect=AssertionError("task enqueued")):
        client = TestClient(app)
        response = client.post(
            path,
            content=b"malformed multipart body",
            headers={"Content-Type": "multipart/form-data; boundary=missing"},
        )
        image = io.BytesIO()
        Image.new("RGB", (16, 16)).save(image, format="JPEG")
        valid_response = client.post(path, files={"files": ("photo.jpg", image.getvalue(), "image/jpeg")})

    for rejected in (response, valid_response):
        assert rejected.status_code == 503
        assert rejected.json()["error"]["code"] == "BLUR_ASYNC_DISABLED"
        assert rejected.json()["request_id"] == rejected.headers["X-Request-ID"]


def test_blur_async_is_available_by_default():
    from src.config import Settings
    from src.main import create_app

    settings = Settings(MODEL_PROFILE="blur", METRICS_ENABLED=False)
    assert settings.BLUR_ASYNC_ENABLED
    with patch("src.main.get_settings", return_value=settings):
        app = create_app()
    app.state.settings = settings
    response = TestClient(app).post(
        "/api/v1/blur/classify/batch",
        content=b"malformed multipart body",
        headers={"Content-Type": "multipart/form-data; boundary=missing"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("profile", ["all", "blur"])
def test_no_face_or_bib_routes_in_any_profile(profile):
    from fastapi import FastAPI

    from src.api.v1.router import make_v1_router

    # The OpenAPI paths, not router.routes: FastAPI 0.141 (the lock) wraps included
    # routers in objects that have no .path.
    app = FastAPI()
    app.include_router(make_v1_router(profile))
    paths = list(app.openapi()["paths"])
    assert "/blur/detect" in paths, paths
    assert not [p for p in paths if p.startswith(("/faces", "/bibs"))], paths
    # The schema omits include_in_schema=False routes; a request still finds them.
    client = TestClient(app)
    for old in ("/faces/detect", "/faces/search", "/bibs/recognize"):
        assert client.post(old).status_code == 404, old


class TestDockerBuildContext:
    """Inspect the real Docker context for the audited blur artifact."""

    def test_context_carries_only_the_audited_blur_pair(self):
        if not shutil.which("docker"):
            pytest.skip("docker unavailable — cannot inspect the build context")
        if not (PROJECT_ROOT / "models/blur_classifier/people_rule.json").exists():
            pytest.skip("audited local artifact absent — release build must provision it")

        tag = "quickpitik-context-probe:test"
        # busybox rather than the real Dockerfile: this asserts what the context
        # contains, and the ML dependency install would take minutes to tell us
        # the same thing.
        probe = b"FROM busybox\nCOPY . /ctx/\n"
        try:
            build = subprocess.run(
                ["docker", "build", "-q", "-f", "-", "-t", tag, str(PROJECT_ROOT)],
                input=probe,
                capture_output=True,
                timeout=600,
            )
            if build.returncode != 0:
                pytest.fail(f"context probe build failed:\n{build.stderr.decode(errors='replace')}")

            listing = subprocess.run(
                [
                    "docker", "run", "--rm", tag, "sh", "-c",
                    "ls -A /ctx /ctx/models /ctx/models/blur_classifier "
                    "/ctx/models/person_detector",
                ],
                capture_output=True,
                timeout=120,
            )
            out = listing.stdout.decode(errors="replace")
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=120)

        assert "manifest.json" in out, "models/manifest.json never reached the image"
        assert "people_rule.json" in out, "audited rule absent from build context"
        assert "yolov8n_seg_person.onnx" in out, "person detector absent from build context"
        for weight in ("bib_detection", "buffalo_l"):
            assert weight not in out, (
                f"{weight} is in the build context — weights are meant to be "
                f"volume-mounted, and models/* is gitignored so a clean "
                f"checkout could not bake them anyway"
            )
        for heavy in ("Training-Images", ".venv", "runs"):
            assert heavy not in out, f"{heavy} is in the build context"


def test_default_model_profile_is_blur(monkeypatch):
    from src.config import Settings

    monkeypatch.delenv("MODEL_PROFILE", raising=False)
    assert Settings(_env_file=None).MODEL_PROFILE == "blur"


def test_dockerfile_builds_blur_by_default():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.findall(r"^ARG MODEL_PROFILE=(\S+)\s*$", dockerfile, re.M) == ["blur", "blur"]


def test_dev_compose_publishes_ports_on_loopback_only():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    ports = [p for svc in compose["services"].values() for p in svc.get("ports", [])]
    assert len(ports) >= 3
    assert all(
        p.get("host_ip") == "127.0.0.1" if isinstance(p, dict) else str(p).startswith("127.0.0.1:")
        for p in ports
    ), ports


def test_makefile_serves_the_api_on_loopback_only():
    # DEBUG=true turns the key check off, so the host-run dev API must not be
    # reachable from shared Wi-Fi (V-17 follow-up, owner decision D7, 2026-10-16).
    makefile = (PROJECT_ROOT / "Makefile").read_text()
    assert re.findall(r"uvicorn .*--host (\S+)", makefile) == ["127.0.0.1", "127.0.0.1"]


def test_no_face_or_bib_workers_or_model_loaders():
    from src.ml.registry import ModelRegistry
    from src.workers.celery_app import celery_app

    assert not [m for m in celery_app.conf.include if m.endswith(("face_tasks", "bib_tasks"))]
    assert not [r for r in celery_app.conf.task_routes if r.startswith(("faces.", "bibs."))]
    for loader in ("_load_face", "_load_bib_detector", "_load_bib_ocr"):
        assert not hasattr(ModelRegistry, loader), loader


def test_install_and_manifest_have_no_face_or_bib():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    for lib in ("insightface", "paddleocr", "paddlepaddle"):
        assert lib not in pyproject, lib
    manifest = json.loads((root / "models" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["models"]) == {"blur_classifier", "person_detector"}
    prod = (root / "docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "celery-face" not in prod and "celery-bib" not in prod


def test_async_railway_layout_keeps_default_queue_off_the_blur_worker():
    layout_file = Path(__file__).resolve().parents[1] / ".railway" / "railway-async.ts"
    layout = layout_file.read_text(encoding="utf-8")
    starts = re.findall(r'start:\s*"([^"]+)"', layout)
    workers = [s for s in starts if " worker " in s]
    assert any("-Q blur " in s for s in workers) and any("-Q default " in s for s in workers)
    assert not [s for s in workers if "blur,default" in s or "default,blur" in s]


def test_supply_chain_is_pinned():
    root = PROJECT_ROOT
    lock = (root / "requirements-blur.lock").read_text(encoding="utf-8")
    pins = re.findall(r"^([A-Za-z0-9_.\-\[\]]+==\S+)", lock, re.M)
    assert pins and len(re.findall(r"--hash=sha256:", lock)) >= len(pins)
    gitlab = (root / ".gitlab-ci.yml").read_text(encoding="utf-8")
    images = re.findall(r"^\s*(?:image|-\s*name):\s*(\S+)", gitlab, re.M)
    images += re.findall(r"^\s*-\s*(docker:\S+)", gitlab, re.M)
    assert images and all("@sha256:" in i for i in images), images
    assert not re.search(r"docker login[^\n]*\s-p\s", gitlab)
    workflow_file = root.parent / ".github" / "workflows" / "ai-api-blur.yml"
    refs = re.findall(r"uses:\s*\S+@(\S+)", workflow_file.read_text(encoding="utf-8"))
    assert refs and all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs), refs
