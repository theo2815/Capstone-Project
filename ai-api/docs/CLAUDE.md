# EventAI - AI API

## Project Overview

This is a modular AI API built with FastAPI (Python 3.11+) for computer vision tasks: **Blur Detection**, **Face Recognition**, and **Bib Number OCR**. Optional C++ acceleration via pybind11.

## Architecture

**4-layer separation. Never skip layers.**

```
src/api/       → HTTP controllers (thin, no logic)
src/services/  → Business logic and orchestration
src/ml/        → ML model wrappers (no HTTP, no DB awareness)
src/db/        → Database models, repositories, session management
```

- `api/` calls `services/`, services call `ml/` and `db/`. Never import `db/` directly from `api/`.
- `ml/` modules must never import from `api/`, `services/`, or `db/`.
- `schemas/` (Pydantic models) are shared across layers for request/response types.

## Key Patterns

### Model Registry

All ML models are loaded **once** at startup via `src/ml/registry.py` and stored in `app.state.model_registry`. Never instantiate models inside route handlers or services. Always get them from the registry:

```python
registry = request.app.state.model_registry
detector = registry.get("blur")
```

### API Response Envelope

Every endpoint returns `src/schemas/common.APIResponse`:

```python
return APIResponse(
    success=True,
    request_id=getattr(request.state, "request_id", ""),
    data=result.model_dump(),
)
```

On errors:
```python
return APIResponse(
    success=False,
    request_id=getattr(request.state, "request_id", ""),
    error={"code": "ERROR_CODE", "message": "Human-readable message"},
)
```

### C++ Fallback Pattern

Any code that uses the C++ extension must use the try/except import pattern:

```python
try:
    from _eventai_cpp import some_function
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
```

Then branch on `_HAS_CPP`. Pure Python/NumPy fallback must always exist.

### Database Access

Use async SQLAlchemy sessions via `src/db/session.get_session()`. Repositories handle CRUD. Face embedding search uses pgvector cosine distance (`<=>` operator).

### Authentication

All endpoints except health checks use `Depends(verify_api_key)` from `src/middleware/auth.py`. In debug mode, missing keys are allowed.

## File Locations

| To add... | Put it in... |
|---|---|
| New API endpoint | `src/api/v1/<feature>.py` then register in `src/api/v1/router.py` |
| New request/response model | `src/schemas/<feature>.py` |
| New business logic | `src/services/<feature>_service.py` |
| New ML model wrapper | `src/ml/<feature>/` with `__init__.py` |
| New database table | `src/db/models.py` then create Alembic migration |
| New repository (CRUD) | `src/db/repositories/<feature>_repo.py` |
| New Celery task | `src/workers/tasks/<feature>_tasks.py` |
| New middleware | `src/middleware/<name>.py` then register in `src/main.py` |
| New config variable | Add to `src/config.py` Settings class and `.env.example` |

## Conventions

- **Python 3.11+** features are allowed (`type | None`, `list[str]`, `match` statements).
- Use `from __future__ import annotations` at the top of every module.
- All route handlers are `async`. CPU-bound ML inference runs via `asyncio.to_thread()` when called from async context.
- API endpoints are versioned: `/api/v1/...`. Future breaking changes go in `src/api/v2/`.
- Logging: use `from src.utils.logging import get_logger; logger = get_logger(__name__)`. Always structured (key=value), never f-string log messages.
- Exceptions: use custom types from `src/utils/exceptions.py`, never raw `HTTPException` in services/ml layers.
- Image validation: always call `validate_and_decode()` from `src/utils/image_utils.py` before processing uploads. Never trust Content-Type headers alone.
- Database: use `UUID` primary keys everywhere. Timestamps use `DateTime(timezone=True)`.
- Environment config: never hardcode values. Add to `src/config.py` and read from env vars.

## Running

```bash
# Install
pip install -e ".[dev]"

# Start infrastructure
docker compose up db redis -d

# Run migrations
alembic upgrade head

# Seed dev data
python scripts/seed_db.py

# Start dev server
make dev

# Run tests
make test

# Lint
make lint
```

## Testing

- Tests live in `tests/unit/`, `tests/integration/`, `tests/e2e/`.
- Test fixtures (images, embeddings) go in `tests/fixtures/`.
- Shared fixtures are in `tests/conftest.py`.
- Use `pytest-asyncio` for async tests. Config: `asyncio_mode = "auto"` in pyproject.toml.

## Dependencies

- **Web**: FastAPI, Uvicorn, Pydantic v2
- **Image**: OpenCV (headless), NumPy, Pillow
- **ML**: InsightFace (RetinaFace + ArcFace), ONNX Runtime, PaddleOCR, Ultralytics YOLOv8
- **DB**: PostgreSQL 16 + pgvector, SQLAlchemy 2 (async), Alembic, asyncpg
- **Queue**: Celery + Redis
- **Auth**: API keys (SHA-256 hashed), python-jose (JWT for future)
- **Observability**: structlog, prometheus-client

Do not add new dependencies without justification. Prefer existing libraries over new ones.

## Things to Avoid

- Never load ML models inside request handlers. Use the registry.
- Never store uploaded images to disk or database. Process in memory, discard after.
- Never log image data or embeddings. Log only request IDs, endpoints, and timings.
- Never commit `.env` files or API keys.
- Never import across layers incorrectly (e.g., `api/` importing from `db/` directly).
- Never use synchronous database calls. Always use async SQLAlchemy.
- Never hardcode thresholds, URLs, or secrets. Use `src/config.py`.
