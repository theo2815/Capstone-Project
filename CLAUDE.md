# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EventAI** — a marathon photography ecosystem with AI-powered blur detection, face recognition, and bib number OCR. The monorepo has four planned products; only `ai-api` is built so far.

| Product | Stack | Status |
|---------|-------|--------|
| `ai-api/` | FastAPI + Celery (Python 3.12) | Phases 1-6 complete |
| `backend/` | Spring Boot (Java) | Not started |
| `website/` | Next.js | Not started |
| `mobile/` | Kotlin (Android) | Not started |

## Build & Run Commands (ai-api)

All commands run from `ai-api/`.

```bash
# Setup
pip install -e ".[dev]"
docker compose up db redis -d
alembic upgrade head
python scripts/seed_db.py        # creates test API key

# Dev server + worker
make dev                          # FastAPI on :8000 with reload
make celery                       # Celery worker (concurrency=2)

# Testing
make test                         # pytest tests/ -v --tb=short
make test-cov                     # + coverage HTML report
pytest tests/unit/test_blur_detector.py -v   # single test file

# Lint & format
make lint                         # ruff check + mypy
make format                       # ruff auto-format + fix

# Database migrations
make migrate                      # alembic upgrade head
make migration msg="description"  # create new migration

# Docker (full stack)
make docker-up                    # build & start all services
make docker-down
make docker-gpu                   # with GPU support
```

## CI Pipeline (GitLab)

Defined in `ai-api/.gitlab-ci.yml`. Stages: `test` then `build`.
- **lint**: `ruff check src/ tests/`
- **typecheck**: `mypy src/` (allowed to fail)
- **unit-tests**: `pytest tests/unit/` with 50% coverage minimum
- **build-image**: Docker build + push (main branch only)

## Architecture (ai-api)

**4-layer separation — never skip layers:**

```
src/api/v1/    → HTTP controllers (thin, no business logic)
src/services/  → Business logic, orchestration
src/ml/        → ML model wrappers (no HTTP, no DB awareness)
src/db/        → Models, repositories, session management
```

Import rules:
- `api/` calls `services/`, services call `ml/` and `db/`
- Never import `db/` directly from `api/`
- `ml/` modules must never import from `api/`, `services/`, or `db/`
- `schemas/` (Pydantic models) are shared across layers

### Model Registry (Singleton)

All ML models load once at startup via `src/ml/registry.py` → `app.state.model_registry`. Never instantiate models in route handlers.

```python
registry = request.app.state.model_registry
detector = registry.get("blur")
```

### Async Pattern

FastAPI handlers are `async`. CPU-bound ML inference runs via `asyncio.to_thread()`. Celery workers use sync DB sessions (`src/db/sync_session.py`) because asyncpg cannot run inside Celery.

### API Response Envelope

Every endpoint returns `src/schemas/common.APIResponse`:
```python
return APIResponse(success=True, request_id=..., data=result.model_dump())
```

### C++ Extensions (Optional)

pybind11 extensions in `src/cpp/`. Always use the try/except fallback pattern:
```python
try:
    from _eventai_cpp import some_function
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
```
Pure Python/NumPy fallback must always exist.

### Batch Processing Pattern

All batch endpoints: accept multipart files → create Job in DB → base64-encode and queue Celery task → return 202 with `job_id` → poll `GET /api/v1/jobs/{job_id}` → webhook on completion.

## Key Conventions

- Python 3.11+ features allowed (`type | None`, `list[str]`, `match`)
- `from __future__ import annotations` at the top of every module
- Logging: `from src.utils.logging import get_logger; logger = get_logger(__name__)` — structured key=value, never f-strings in log messages
- Exceptions: use custom types from `src/utils/exceptions.py`, never raw `HTTPException` in services/ml layers
- Image uploads: always call `validate_and_decode()` from `src/utils/image_utils.py`
- Database: UUID primary keys, `DateTime(timezone=True)` timestamps
- Auth: `Depends(verify_api_key)` on all endpoints except health; debug mode bypasses
- Config: never hardcode values — add to `src/config.py` Settings class and `.env.example`
- Ruff: line length 100, target Python 3.11
- Tests: `tests/unit/`, `tests/integration/`, `tests/e2e/`; `asyncio_mode = "auto"` in pyproject.toml

## Things to Avoid

- Loading ML models inside request handlers (use the registry)
- Storing uploaded images to disk or DB (process in memory, discard)
- Logging image data or embeddings (log only request IDs, endpoints, timings)
- Synchronous DB calls in FastAPI handlers (Celery workers are the exception — they must use sync sessions)
- Cross-layer imports that skip the layering (e.g., `api/` → `db/`)
- Adding dependencies without justification

## Where to Put New Code

| Adding... | Location |
|-----------|----------|
| API endpoint | `src/api/v1/<feature>.py`, register in `src/api/v1/router.py` |
| Request/response model | `src/schemas/<feature>.py` |
| Business logic | `src/services/<feature>_service.py` |
| ML model wrapper | `src/ml/<feature>/` |
| Database table | `src/db/models.py`, then create Alembic migration |
| Repository (CRUD) | `src/db/repositories/<feature>_repo.py` |
| Celery task | `src/workers/tasks/<feature>_tasks.py` |
| Middleware | `src/middleware/<name>.py`, register in `src/main.py` |
| Config variable | `src/config.py` Settings class + `.env.example` |

## ML Training Pipelines

```bash
# Blur classifier
python scripts/prepare_blur_dataset.py
python scripts/train_blur_classifier.py
python scripts/export_blur_classifier.py

# Face + bib detector
python scripts/auto_annotate_face_bib.py
python scripts/train_face_bib_detector.py
python scripts/export_face_bib_detector.py
```

Training images in `Training-Images/` (gitignored). Model artifacts in `models/` (gitignored except manifests). See `ai-api/docs/` for detailed training guides.

## Infrastructure

| Service | Purpose |
|---------|---------|
| PostgreSQL 16 + pgvector | Jobs, webhooks, face embeddings (cosine search via `<=>`) |
| Redis 7.4 | Celery broker/backend, rate limiting, API key cache |
| Celery worker | Async batch processing (queues: default, blur, face, bib) |

Start infra: `docker compose up db redis -d`

## Detailed ai-api Docs

The `ai-api/docs/CLAUDE.md` file has more granular guidance including API endpoint details, ML feature specifics, and the full file location reference. The `ai-api/docs/` directory contains architecture docs, API reference, deployment guides, and training plans.
