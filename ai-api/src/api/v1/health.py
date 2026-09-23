from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.middleware.auth import verify_api_key
from src.schemas.common import APIResponse, HealthResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def liveness(request: Request) -> HealthResponse:
    """Liveness probe. Returns 200 if the process is alive.

    Exposes only version — no environment details.
    """
    settings = request.app.state.settings
    return HealthResponse(
        status="alive",
        version=settings.APP_VERSION,
    )


async def _checks(request: Request) -> ReadinessResponse:
    registry = request.app.state.model_registry
    from src.db.session import check_db_health

    redis = getattr(request.app.state, "redis", None)
    timeout = request.app.state.settings.READINESS_TIMEOUT_SECONDS

    async def db_healthy() -> bool:
        try:
            return await asyncio.wait_for(check_db_health(), timeout)
        except Exception:
            return False

    async def redis_healthy() -> bool:
        if redis is None:
            return False
        try:
            from src.middleware.rate_limit import TOKEN_BUCKET_SCRIPT

            async def probe() -> bool:
                await redis.ping()
                await redis.set("apikey:readyz", "ready", ex=10)
                if await redis.get("apikey:readyz") != "ready":
                    return False
                await redis.eval(TOKEN_BUCKET_SCRIPT, 1, "ratelimit:readyz", 1, 1, time.time())
                return True

            return await asyncio.wait_for(probe(), timeout)
        except Exception:
            return False

    db_ok, redis_ok = await asyncio.gather(db_healthy(), redis_healthy())
    from src.utils.timeout import inference_healthy

    return ReadinessResponse(
        models_loaded=registry.all_loaded(),
        models=registry.status(),
        inference=(
            request.app.state.settings.MODEL_PROFILE != "blur" or inference_healthy()
        ),
        database=db_ok,
        redis=redis_ok,
    )


@router.get("/readyz", include_in_schema=False)
async def public_readiness(request: Request) -> JSONResponse:
    """Minimal public readiness for the deployment probe."""
    checks = await _checks(request)
    healthy = checks.models_loaded and checks.inference and checks.database and checks.redis
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ready" if healthy else "unavailable"},
    )


@router.get("/ready", response_model=APIResponse)
async def readiness(
    request: Request,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Authenticated detailed readiness."""
    checks = await _checks(request)
    healthy = checks.models_loaded and checks.inference and checks.database and checks.redis

    response_data = APIResponse(
        success=healthy,
        request_id="healthcheck",
        data=checks.model_dump(),
    )

    if not healthy:
        return JSONResponse(
            status_code=503,
            content=response_data.model_dump(mode="json"),
        )

    return response_data
