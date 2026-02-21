from __future__ import annotations

from fastapi import APIRouter, Request

from src.schemas.common import APIResponse, HealthResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def liveness(request: Request) -> HealthResponse:
    """Liveness probe. Returns 200 if the process is alive."""
    settings = request.app.state.settings
    return HealthResponse(
        status="alive",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get("/ready", response_model=APIResponse)
async def readiness(request: Request) -> APIResponse:
    """Readiness probe. Checks that models, DB, and Redis are available."""
    registry = request.app.state.model_registry
    from src.db.session import check_db_health

    db_ok = await check_db_health()

    redis = getattr(request.app.state, "redis", None)
    redis_ok = False
    if redis:
        try:
            await redis.ping()
            redis_ok = True
        except Exception:
            redis_ok = False

    checks = ReadinessResponse(
        models_loaded=registry.all_loaded(),
        database=db_ok,
        redis=redis_ok,
    )
    healthy = checks.models_loaded and checks.database and checks.redis

    return APIResponse(
        success=healthy,
        request_id="healthcheck",
        data=checks.model_dump(),
    )
