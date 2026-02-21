from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

    # Initialize logging
    setup_logging(settings.LOG_LEVEL)

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize Redis
    try:
        import redis.asyncio as aioredis

        app.state.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
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

    yield

    # Shutdown
    logger.info("Shutting down...")
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

    # Middleware (order matters: last added = first executed)
    app.add_middleware(RequestIDMiddleware)
    setup_cors(app, settings.ALLOWED_ORIGINS)

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

    return app
