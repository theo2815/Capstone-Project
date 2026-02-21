from __future__ import annotations

from fastapi import APIRouter

from src.api.v1.bibs import router as bibs_router
from src.api.v1.blur import router as blur_router
from src.api.v1.faces import router as faces_router
from src.api.v1.health import router as health_router
from src.api.v1.jobs import router as jobs_router
from src.api.v1.webhooks import router as webhooks_router

v1_router = APIRouter()

v1_router.include_router(health_router)
v1_router.include_router(blur_router)
v1_router.include_router(faces_router)
v1_router.include_router(bibs_router)
v1_router.include_router(jobs_router)
v1_router.include_router(webhooks_router)
