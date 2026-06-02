from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "QuickPitik API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    SQL_ECHO: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2
    MAX_REQUEST_BODY: int = 500 * 1024 * 1024  # 500 MB (covers stream batch of 500 images)

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/quickpitik"

    # Redis (use redis://:password@host:port/db for authenticated instances)
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Models
    MODEL_DIR: str = "./models"
    USE_GPU: bool = False
    GPU_DEVICE: int = 0

    # Blur Detection
    BLUR_THRESHOLD: float = 100.0
    BLUR_DETECTION_MIN_CONFIDENCE: float = 0.5

    # Face Recognition
    FACE_SIMILARITY_THRESHOLD: float = 0.4
    FACE_DET_SIZE: int = 640
    FACE_MIN_ENROLLMENT_CONFIDENCE: float = 0.7

    # Bib Recognition
    BIB_MIN_CHARS: int = 2

    # Inference
    INFERENCE_TIMEOUT: int = 120  # Per-image inference timeout in seconds
    INFERENCE_BATCH_TIMEOUT: int = 300  # Max seconds for any single batch ONNX call
    INFERENCE_SUB_BATCH_SIZE: int = 50  # Match MAX_BATCH_SIZE: 1 ONNX call per task
    ONNX_INTRA_OP_THREADS: int = 6
    ONNX_INTER_OP_THREADS: int = 4

    # OCR
    OCR_MAX_WORKERS: int = 8  # Thread pool size for PaddleOCR batch inference

    # Auth
    API_KEY_HEADER: str = "X-API-Key"
    JWT_PUBLIC_KEY: str = ""
    JWT_ALGORITHM: str = "RS256"

    # Rate Limiting
    RATE_LIMIT_DEFAULT: int = 60
    RATE_LIMIT_BURST: int = 10

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # Webhooks
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_MAX_RETRIES: int = 3
    WEBHOOK_SECRET_KEY: str = ""  # Fernet key for encrypting webhook secrets at rest

    # Celery
    CELERY_SECURITY_KEY: str = ""

    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    MAX_BATCH_SIZE: int = 50  # Raised from 20: Redis now 1GB + lazy decode keeps memory O(1)
    MEGA_BATCH_MAX_SIZE: int = 500  # Max images per mega-batch request (server-side chunking)
    STREAM_BATCH_MAX_SIZE: int = 500  # Max images per streaming sync batch request
    STREAM_CONCURRENCY: int = 8  # Thread pool size for concurrent image processing
    STREAM_CLASSIFY_MAX_SIZE: int = 500  # Max images per streaming classify request
    STREAM_CLASSIFY_CONCURRENCY: int = 8  # Thread pool for classify/stream (ONNX is heavier)
    MAX_ACTIVE_JOBS_PER_KEY: int = 10  # Backpressure: max pending+processing jobs per API key
    # Internal-tier callers (e.g. the Spring backend's bulk photo-indexing
    # drain) submit a FACE + BIB job per event-batch and may drain several
    # events at once, so they get a higher ceiling than the public default.
    MAX_ACTIVE_JOBS_PER_KEY_INTERNAL: int = 50
    JOB_RETENTION_DAYS: int = 7  # Auto-delete completed/failed jobs older than this

    # Blob store — images written to shared volume instead of base64-in-Redis
    BLOB_STORE_PATH: str = "/tmp/quickpitik-blobs"

    # Image preprocessing — downscale large images before inference.
    # 640 was too aggressive for marathon photos: a 2400x1600 frame got shrunk
    # to 426x640, leaving the bib at ~40x27px which is below the YOLO bib
    # detector's effective range. 1280 keeps inference fast while preserving
    # enough bib pixels for OCR (~80x54px on a typical race shot).
    MAX_INFERENCE_DIMENSION: int = 1280

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
