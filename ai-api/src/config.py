from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "EventAI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    SQL_ECHO: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2
    MAX_REQUEST_BODY: int = 50 * 1024 * 1024  # 50 MB (covers batch of 5 x 10MB)

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/eventai"

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
    MAX_BATCH_SIZE: int = 20  # Kept low to limit memory (base64 in Redis)
    MAX_ACTIVE_JOBS_PER_KEY: int = 10  # Backpressure: max pending+processing jobs per API key

    # Image preprocessing — downscale large images before inference
    # Models resize internally to 640x640 so images beyond this are wasted memory
    MAX_INFERENCE_DIMENSION: int = 2048  # 0 = disabled

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
