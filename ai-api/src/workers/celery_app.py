from __future__ import annotations

from celery import Celery

from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "eventai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["src.workers.tasks"])

# Register worker startup signals (model loading, sync DB init)
import src.workers.model_loader  # noqa: F401, E402
