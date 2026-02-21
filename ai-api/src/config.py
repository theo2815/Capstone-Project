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

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/eventai"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Models
    MODEL_DIR: str = "./models"
    USE_GPU: bool = False
    GPU_DEVICE: int = 0

    # Blur Detection
    BLUR_THRESHOLD: float = 100.0

    # Face Recognition
    FACE_SIMILARITY_THRESHOLD: float = 0.4
    FACE_DET_SIZE: int = 640

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

    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    MAX_BATCH_SIZE: int = 100

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
