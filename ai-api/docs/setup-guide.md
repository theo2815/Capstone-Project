# Setup Guide

## Prerequisites

- **Python 3.11 or 3.12** (not 3.10 or earlier, not 3.13)
- **Docker Desktop** (for PostgreSQL and Redis)
- **Git**

## Quick Start

### 1. Clone and enter the project
```bash
cd "c:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api"
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# Install the project with dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### 4. Set up environment variables
```bash
# Copy the template
copy .env.example .env

# Edit .env with your settings (the defaults work for local Docker setup)
```

### 5. Start PostgreSQL and Redis
```bash
docker compose up db redis -d
```
This starts:
- PostgreSQL (with pgvector) on port 5432
- Redis on port 6379

### 6. Run database migrations
```bash
# First time: create the initial migration
alembic revision --autogenerate -m "initial tables"

# Apply migrations
alembic upgrade head
```

### 7. Seed the database (optional)
```bash
python scripts/seed_db.py
```
This creates a development API key: `sk_dev_eventai_test_key_12345`

### 8. Start the API server
```bash
# Development mode (auto-reloads on code changes)
make dev

# Or manually:
uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 9. Verify it works
Open your browser to: http://localhost:8000/docs

This shows the Swagger UI with all endpoints. You can test them interactively.

Or via curl:
```bash
# Health check (no auth needed)
curl http://localhost:8000/api/v1/health

# Readiness check
curl http://localhost:8000/api/v1/health/ready

# Test blur detection (needs API key)
curl -X POST http://localhost:8000/api/v1/blur/detect \
  -H "X-API-Key: sk_dev_eventai_test_key_12345" \
  -F "file=@some_photo.jpg"
```

---

## Running Everything with Docker

If you don't want to install Python locally, run the entire stack in Docker:

```bash
# Build and start all services
docker compose up --build

# Or in detached mode
docker compose up --build -d

# View logs
docker compose logs -f ai-api

# Stop everything
docker compose down
```

This starts 4 containers: ai-api, celery-worker, PostgreSQL, Redis.

---

## Common Commands

| Command | What it does |
|---|---|
| `make dev` | Start API server with auto-reload |
| `make run` | Start API server in production mode |
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage report |
| `make lint` | Check code style and types |
| `make format` | Auto-format code |
| `make migrate` | Apply database migrations |
| `make migration msg="add new table"` | Create a new migration |
| `make docker-up` | Start all Docker services |
| `make docker-down` | Stop all Docker services |
| `make celery` | Start Celery worker locally |
| `make install` | Install dev dependencies |
| `make clean` | Remove cached files |

---

## Environment Variables

All configuration is done via environment variables. See `.env.example` for the full list.

**Key variables:**

| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `false` | Enable debug mode (shows Swagger UI, relaxes auth) |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `USE_GPU` | `false` | Enable GPU inference |
| `BLUR_THRESHOLD` | `100.0` | Default Laplacian variance threshold |
| `FACE_SIMILARITY_THRESHOLD` | `0.4` | Minimum cosine similarity for face match |
| `MAX_FILE_SIZE` | `10485760` | Maximum upload size in bytes (10MB) |
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]` | CORS allowed origins |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Troubleshooting

### "Database not initialized" error
Make sure PostgreSQL is running: `docker compose up db -d`

### "Redis not available" warning
Not critical for development. Redis is optional. Rate limiting and caching will be disabled.

### Models fail to load
On first run, InsightFace and PaddleOCR download models automatically (~500MB total). Ensure you have internet access. The YOLOv8 bib detector requires a custom-trained model (see Phase 4).

### Port 8000 already in use
Either stop the other process or change the port:
```bash
uvicorn src.main:create_app --factory --port 8080 --reload
```

### Import errors after install
Make sure your virtual environment is activated and you installed with `pip install -e ".[dev]"`.
