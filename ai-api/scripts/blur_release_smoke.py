"""Exercise a clean blur image against disposable Postgres/Redis and a holdout.

The release job mounts this script into the built image. It creates short-lived
keys in the disposable database and never prints their values.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import secrets
import subprocess
import sys
import uuid
from pathlib import Path

import asyncpg
import httpx
from PIL import Image


async def run(smoke_only: bool) -> None:
    base = os.environ.get("BLUR_SMOKE_URL", "http://127.0.0.1:8000")
    holdout = os.environ.get("BLUR_HOLDOUT_DIR", "")
    if not smoke_only and (not holdout or not Path(holdout).is_dir()):
        raise RuntimeError("BLUR_HOLDOUT_DIR must contain the approved labelled holdout")
    limits = {}
    if not smoke_only:
        for name in ("BLUR_MIN_ACCURACY", "BLUR_MAX_MISS_RATE", "BLUR_MAX_OVERCULL_RATE"):
            try:
                value = float(os.environ[name])
            except (KeyError, ValueError):
                raise RuntimeError(f"{name} must be set to a number from 0 to 1") from None
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError(f"{name} must be set to a number from 0 to 1")
            limits[name] = str(value)

    desktop = secrets.token_urlsafe(32)
    operator = secrets.token_urlsafe(32)
    ids = [uuid.uuid4(), uuid.uuid4()]
    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = await asyncpg.connect(dsn)
    try:
        for key_id, key, scope in (
            (ids[0], desktop, '["blur:read","jobs:read"]'),
            (ids[1], operator, '["metrics:read"]'),
        ):
            await conn.execute(
                """INSERT INTO api_keys (id, key_hash, name, scopes, rate_tier, active)
                   VALUES ($1, $2, 'Release smoke', $3::jsonb, 'internal', true)""",
                key_id, hashlib.sha256(key.encode()).hexdigest(), scope,
            )

        async with httpx.AsyncClient(base_url=base, timeout=20) as client:
            for _ in range(90):
                try:
                    ready = await client.get("/api/v1/health/readyz")
                    if ready.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("Blur image did not become ready")

            image = Image.new("RGB", (64, 64), color=(40, 80, 120))
            data = io.BytesIO()
            image.save(data, format="JPEG")
            response = await client.post(
                "/api/v1/blur/classify",
                headers={"X-API-Key": desktop},
                files={"file": ("smoke.jpg", data.getvalue(), "image/jpeg")},
            )
            if response.status_code != 200 or not response.json().get("success"):
                raise RuntimeError(f"Real-model classify smoke failed: HTTP {response.status_code}")
            for path in (
                "/api/v1/blur/detect/batch",
                "/api/v1/blur/classify/batch",
                "/api/v1/blur/detect/mega",
                "/api/v1/blur/classify/mega",
            ):
                blocked = await client.post(path, headers={"X-API-Key": desktop})
                if (
                    blocked.status_code != 503
                    or blocked.json().get("error", {}).get("code") != "BLUR_ASYNC_DISABLED"
                ):
                    raise RuntimeError(f"Async blur route is enabled: {path}")
            if (await client.get("/metrics", headers={"X-API-Key": desktop})).status_code != 403:
                raise RuntimeError("Desktop key can read operational metrics")
            metrics = await client.get("/metrics", headers={"X-API-Key": operator})
            samples = metrics.text.splitlines()
            if metrics.status_code != 200 or not all((
                any(line.startswith("blur_model_info{") and line.endswith(" 1.0")
                    for line in samples),
                any(line.startswith('blur_images_total{operation="classify",outcome="succeeded"} ')
                    for line in samples),
                any(line.startswith('blur_dependency_ready{dependency="models"} 1.0')
                    for line in samples),
                any(line.startswith("process_resident_memory_bytes ")
                    and float(line.rsplit(" ", 1)[-1]) > 0 for line in samples),
            )):
                raise RuntimeError("Operational metrics are missing")

        if not smoke_only:
            env = {**os.environ, "AI_API_KEY": desktop}
            result = subprocess.run(
                [sys.executable, "/app/scripts/blur_gate.py", holdout,
                 "--url", base, "--report-json", "/reports/blur-gate-report.json",
                 "--min-accuracy", limits["BLUR_MIN_ACCURACY"],
                 "--max-miss-rate", limits["BLUR_MAX_MISS_RATE"],
                 "--max-overcull-rate", limits["BLUR_MAX_OVERCULL_RATE"]],
                env=env, capture_output=True, text=True, check=False,
            )
            report = json.loads(Path("/reports/blur-gate-report.json").read_text(encoding="utf-8"))
            print("Holdout gate:", report["status"], report["counts"], report["rates"])
            if result.returncode or report.get("status") != "PASS":
                raise RuntimeError("Holdout gate failed; inspect the private report")
        print("Blur release smoke passed" + (" (holdout gate included)" if not smoke_only else ""))
    finally:
        await conn.execute("DELETE FROM api_keys WHERE id = ANY($1::uuid[])", ids)
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.smoke_only))
