from __future__ import annotations

from fastapi import Depends, Request

from src.middleware.auth import verify_api_key
from src.ml.registry import ModelRegistry


def get_model_registry(request: Request) -> ModelRegistry:
    """Dependency: get the model registry from app state."""
    return request.app.state.model_registry


def get_settings(request: Request):
    """Dependency: get application settings from app state."""
    return request.app.state.settings


async def get_redis(request: Request):
    """Dependency: get the Redis client from app state."""
    return getattr(request.app.state, "redis", None)


async def require_auth(key_meta: dict = Depends(verify_api_key)) -> dict:
    """Dependency: require valid API key authentication."""
    return key_meta
