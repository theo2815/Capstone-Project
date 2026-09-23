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

import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROD_COMPOSE = PROJECT_ROOT / "docker-compose.prod.yml"
GPU_COMPOSE = PROJECT_ROOT / "docker-compose.gpu.yml"

# The API plus every Celery worker. All five load models and all five decrypt
# webhook secrets, so anything below that says "all app services" means these.
APP_SERVICES = (
    "ai-api",
    "celery-blur",
    "celery-face",
    "celery-bib",
    "celery-default",
)


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
        assert "!models/blur_classifier/blur_classifier.onnx" in lines
        assert "!models/blur_classifier/class_names.json" in lines
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
        registry = self._registry(
            blur=True, face=True, bib_ocr=True, blur_classifier=False, bib_detector=False
        )
        assert registry.status() == {
            "blur": True,
            "face": True,
            "bib_ocr": True,
            "blur_classifier": False,
            "bib_detector": False,
        }

    def test_the_gap_status_exists_to_expose(self):
        """Not a redundant assertion — this pins the exact hole. The verdict
        stays green with both optional models missing, which is why the payload
        needs the per-model detail alongside it. If someone later makes them
        required, this failing is the intended signal to update the docs and the
        deployment guide, not to delete the test.
        """
        registry = self._registry(
            blur=True, face=True, bib_ocr=True, blur_classifier=False, bib_detector=False
        )
        assert registry.all_loaded() is True
        assert registry.status()["blur_classifier"] is False

    def test_a_missing_required_model_still_fails_the_verdict(self):
        registry = self._registry(
            blur=True, face=False, bib_ocr=True, blur_classifier=True, bib_detector=True
        )
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
        calls = 0

        async def ping(self):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("redis unavailable")

        async def aclose(self):
            pass

    redis = Redis()
    monkeypatch.setattr("redis.asyncio.from_url", lambda *_args, **_kwargs: redis)
    with TestClient(create_app()) as client:
        assert client.get("/api/v1/health/readyz").status_code == 200


@pytest.mark.asyncio
async def test_blur_profile_loads_only_blur_models(monkeypatch):
    from src.ml.registry import ModelRegistry

    registry = ModelRegistry()
    called = []
    for name in ("blur", "blur_classifier", "face", "bib_detector", "bib_ocr"):
        monkeypatch.setattr(
            registry, f"_load_{name}",
            lambda _settings, n=name: called.append(n) or (
                SimpleNamespace(session=object()) if n == "blur_classifier" else object()
            )
        )
    monkeypatch.setattr(
        registry, "_preimport_libs", lambda: pytest.fail("full ML libraries imported")
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


def test_blur_profile_does_not_import_face_or_bib_routes():
    code = (
        "import os, sys\n"
        "os.environ['MODEL_PROFILE'] = 'blur'\n"
        "from src.main import create_app\n"
        "create_app()\n"
        "assert 'src.api.v1.faces' not in sys.modules\n"
        "assert 'src.api.v1.bibs' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


class TestDockerBuildContext:
    """Inspect the real Docker context for the audited blur artifact."""

    def test_context_carries_only_the_audited_blur_pair(self):
        if not shutil.which("docker"):
            pytest.skip("docker unavailable — cannot inspect the build context")
        if not (PROJECT_ROOT / "models/blur_classifier/blur_classifier.onnx").exists():
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
                    "ls -A /ctx /ctx/models /ctx/models/blur_classifier",
                ],
                capture_output=True,
                timeout=120,
            )
            out = listing.stdout.decode(errors="replace")
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=120)

        assert "manifest.json" in out, "models/manifest.json never reached the image"
        assert "blur_classifier.onnx" in out, "audited ONNX absent from build context"
        assert "class_names.json" in out, "audited labels absent from build context"
        for weight in ("bib_detection", "buffalo_l"):
            assert weight not in out, (
                f"{weight} is in the build context — weights are meant to be "
                f"volume-mounted, and models/* is gitignored so a clean "
                f"checkout could not bake them anyway"
            )
        for heavy in ("Training-Images", ".venv", "runs"):
            assert heavy not in out, f"{heavy} is in the build context"
