from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from starlette.middleware.base import BaseHTTPMiddleware

from src.api.v1.router import v1_router
from src.config import get_settings
from src.db.session import close_db, init_db
from src.middleware.cors import setup_cors
from src.middleware.request_id import RequestIDMiddleware
from src.ml.registry import ModelRegistry
from src.utils.exceptions import EventAIError
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    app.state.settings = settings

    # Prevent DEBUG mode in production
    if settings.DEBUG and settings.ENVIRONMENT == "production":
        raise RuntimeError(
            "FATAL: DEBUG=true is not allowed when ENVIRONMENT=production. "
            "Set DEBUG=false in your .env or environment variables."
        )

    # Initialize logging
    setup_logging(settings.LOG_LEVEL)

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize Redis with connection pooling
    try:
        import redis.asyncio as aioredis

        app.state.redis = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=50,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        await app.state.redis.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis not available, running without cache", error=str(e))
        app.state.redis = None

    # Load ML models
    registry = ModelRegistry()
    await registry.load_all(settings)
    app.state.model_registry = registry
    logger.info("ML models loaded", all_loaded=registry.all_loaded())

    # SEC-6: Warn if webhook secrets will be stored as plaintext
    if not settings.WEBHOOK_SECRET_KEY:
        logger.warning(
            "WEBHOOK_SECRET_KEY is not set — webhook secrets will be stored as plaintext. "
            "Set WEBHOOK_SECRET_KEY to a Fernet key to enable encryption."
        )

    yield

    # Shutdown — allow in-flight requests to drain
    import asyncio

    logger.info("Shutting down, waiting for in-flight requests to drain...")
    await asyncio.sleep(2)  # Brief grace period for request completion
    await registry.unload_all()
    if app.state.redis:
        await app.state.redis.aclose()
    await close_db()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Modular AI API for Blur Detection, Face Recognition, and Bib Number OCR",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Request timeout middleware — cancel requests exceeding 60s
    class TimeoutMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            import asyncio

            try:
                return await asyncio.wait_for(call_next(request), timeout=60.0)
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={
                        "success": False,
                        "request_id": getattr(request.state, "request_id", ""),
                        "error": {
                            "code": "REQUEST_TIMEOUT",
                            "message": "Request processing timed out",
                        },
                    },
                )

    # SEC-11: Security response headers (pure ASGI — no response body buffering)
    _security_headers = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"x-xss-protection", b"1; mode=block"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
    ]
    if settings.ENVIRONMENT == "production":
        _security_headers.append(
            (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
        )

    class SecurityHeadersMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return

            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.extend(_security_headers)
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_wrapper)

    # RS-5: Rate limit headers on all responses
    class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            rate_info = getattr(request.state, "rate_info", None)
            if rate_info and rate_info.get("remaining", -1) >= 0:
                response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
            return response

    # Middleware (order matters: last added = outermost = first on request)
    app.add_middleware(RateLimitHeadersMiddleware)
    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(RequestIDMiddleware)
    setup_cors(app, settings.ALLOWED_ORIGINS)
    # SecurityHeaders outermost: catches ALL responses including timeout/rate-limit errors
    app.add_middleware(SecurityHeadersMiddleware)

    # Exception handlers
    @app.exception_handler(EventAIError)
    async def eventai_error_handler(request: Request, exc: EventAIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "request_id": getattr(request.state, "request_id", ""),
                "error": {"code": type(exc).__name__, "message": exc.message},
            },
        )

    # API routes
    app.include_router(v1_router, prefix="/api/v1")

    # Prometheus metrics — auto-instruments all endpoints
    # SEC-4: Metrics exposed at /metrics, protected by API key in production
    from prometheus_fastapi_instrumentator import Instrumentator

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        excluded_handlers=["/metrics"],
    )
    instrumentator.instrument(app)

    if settings.DEBUG:
        # In development, expose /metrics without auth
        instrumentator.expose(app, include_in_schema=False)
    else:
        # In production, serve metrics behind a dedicated authenticated route
        from starlette.responses import Response
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        @app.get("/metrics", include_in_schema=False)
        async def metrics(request: Request):
            import hashlib

            from src.db.session import get_session_ctx

            api_key = request.headers.get(settings.API_KEY_HEADER)
            if not api_key:
                return JSONResponse(
                    status_code=401, content={"detail": "Missing API key"}
                )

            # Validate API key against DB (same logic as verify_api_key)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            from src.db.models import APIKey
            from sqlalchemy import select

            try:
                async with get_session_ctx() as session:
                    result = await session.execute(
                        select(APIKey).where(
                            APIKey.key_hash == key_hash,
                            APIKey.active.is_(True),
                        )
                    )
                    if result.scalar_one_or_none() is None:
                        return JSONResponse(
                            status_code=403, content={"detail": "Invalid API key"}
                        )
            except Exception:
                return JSONResponse(
                    status_code=503, content={"detail": "Cannot validate API key"}
                )

            return Response(
                content=generate_latest(), media_type=CONTENT_TYPE_LATEST
            )

    return app
