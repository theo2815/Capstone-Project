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
from pathlib import Path

import pytest
import yaml

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
    """The image ships manifest.json and nothing else, by design.

    So the host mount is the ONLY way weights reach a container. Without it
    blur_classifier and bib_detector are absent — and because the registry
    calls both optional, /health/ready still answers 200 while every
    /blur/classify* route, the desktop's primary path, fails.
    """

    @pytest.mark.parametrize("name", APP_SERVICES)
    def test_every_app_service_mounts_models(self, name):
        service = _services(PROD_COMPOSE)[name]
        mounts = [m for m in service.get("volumes", []) if ":/app/models" in m]
        assert mounts, f"{name} does not mount ./models — it will run with no weights"
        assert all(m.endswith(":ro") for m in mounts), (
            f"{name} mounts models writable; read-only is what makes a missing "
            f"bundle fail loudly instead of silently re-downloading ~300 MB"
        )

    def test_dockerfile_copies_only_the_manifest(self):
        """Guards the pairing, not the line. `COPY models/` looked correct and
        delivered nothing, because .dockerignore had already excluded it.
        """
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "COPY models/manifest.json" in dockerfile
        assert "COPY models/ " not in dockerfile

    def test_dockerignore_still_excludes_weights_but_keeps_the_manifest(self):
        lines = [
            ln.strip()
            for ln in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        assert "models/" in lines
        assert "!models/manifest.json" in lines
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
            name: (object() if ok else None) for name, ok in loaded.items()
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

    def test_readiness_schema_carries_the_field(self):
        from src.schemas.common import ReadinessResponse

        payload = ReadinessResponse(
            models_loaded=True,
            models={"blur": True, "blur_classifier": False},
            database=True,
            redis=True,
        ).model_dump()
        assert payload["models"]["blur_classifier"] is False


class TestDockerBuildContext:
    """What actually reaches the image.

    Docker and git differ here, and this whole finding turns on the difference:
    git refuses to re-include a file whose parent directory is excluded, Docker
    honours the exception. `models/` + `!models/manifest.json` is therefore
    broken in .gitignore (already fixed there, to `models/*`) and correct in
    .dockerignore. That is too subtle to assert from reading the file, so build
    the real context and look.
    """

    def test_context_carries_the_manifest_and_no_weights(self):
        if not shutil.which("docker"):
            pytest.skip("docker unavailable — cannot inspect the build context")

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
                ["docker", "run", "--rm", tag, "sh", "-c", "ls -A /ctx /ctx/models"],
                capture_output=True,
                timeout=120,
            )
            out = listing.stdout.decode(errors="replace")
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=120)

        assert "manifest.json" in out, "models/manifest.json never reached the image"
        for weight in ("blur_classifier", "bib_detection", "buffalo_l"):
            assert weight not in out, (
                f"{weight} is in the build context — weights are meant to be "
                f"volume-mounted, and models/* is gitignored so a clean "
                f"checkout could not bake them anyway"
            )
        for heavy in ("Training-Images", ".venv", "runs"):
            assert heavy not in out, f"{heavy} is in the build context"
