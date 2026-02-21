from __future__ import annotations

from fastapi import HTTPException, status


class EventAIError(Exception):
    """Base exception for all EventAI errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ImageValidationError(EventAIError):
    """Raised when image validation fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class ModelNotLoadedError(EventAIError):
    """Raised when an ML model is not loaded."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' is not loaded. Check startup logs.",
            status_code=503,
        )


class AuthenticationError(EventAIError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, status_code=401)


class RateLimitExceededError(EventAIError):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after} seconds.",
            status_code=429,
        )


class JobNotFoundError(EventAIError):
    """Raised when an async job is not found."""

    def __init__(self, job_id: str):
        super().__init__(f"Job '{job_id}' not found.", status_code=404)


def eventai_error_to_http(error: EventAIError) -> HTTPException:
    """Convert an EventAI error to a FastAPI HTTPException."""
    return HTTPException(status_code=error.status_code, detail=error.message)
