from __future__ import annotations

from celery import Celery

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

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
    task_reject_on_worker_lost=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    worker_max_tasks_per_child=100,
    broker_connection_retry_on_startup=True,
    # SCALE-1: Route tasks to specialized queues.
    # Start workers with: celery -A src.workers.celery_app worker -Q blur
    # Default queue handles webhooks and any unrouted tasks.
    task_routes={
        "blur.*": {"queue": "blur"},
        "faces.*": {"queue": "face"},
        "bibs.*": {"queue": "bib"},
        "webhooks.*": {"queue": "default"},
        "maintenance.*": {"queue": "default"},
    },
    task_default_queue="default",
    # RS-3/RS-4: Periodic maintenance tasks (Celery beat schedule).
    # Start beat with: celery -A src.workers.celery_app beat
    beat_schedule={
        "reap-stale-jobs": {
            "task": "maintenance.reap_stale_jobs",
            "schedule": 300.0,  # Every 5 minutes
        },
        "cleanup-old-jobs": {
            "task": "maintenance.cleanup_old_jobs",
            "schedule": 86400.0,  # Once per day
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["src.workers.tasks"])

# SEC-3: Enable Celery message signing when security key is configured.
# This uses HMAC to sign task messages, preventing injection via Redis.
# Generate a key with: python -c "import os; print(os.urandom(32).hex())"
_security_key = getattr(settings, "CELERY_SECURITY_KEY", "") or ""
if _security_key:
    celery_app.conf.update(
        security_key=_security_key,
        task_serializer="auth",
        accept_content=["auth", "json"],
        event_serializer="json",
    )
    logger.info("Celery message signing enabled")
else:
    logger.warning(
        "CELERY_SECURITY_KEY not set — task messages are unsigned. "
        "Set CELERY_SECURITY_KEY for production use."
    )

# Register worker startup signals (model loading, sync DB init)
import src.workers.model_loader  # noqa: F401, E402
