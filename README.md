# EventAI

A modular AI-powered REST API for event photography processing. Built with FastAPI (Python 3.11+), PostgreSQL + pgvector, Redis, and Celery.

## Features

- **Blur Detection** -- Laplacian variance + FFT analysis, and a CNN classifier (YOLOv8n-cls, 98.68% accuracy) for 4-class blur type classification
- **Face Recognition** -- InsightFace (RetinaFace + ArcFace) with pgvector cosine similarity search
- **Bib Number OCR** -- PaddleOCR (PP-OCRv5) with optional YOLOv8 bib region detection
- **Async Batch Processing** -- Celery + Redis for processing 100+ images per request
- **Optional C++ Acceleration** -- pybind11 extensions for performance-critical paths (AVX2)

## Project Structure

```
Capstone-Project/
├── ai-api/          # Main application (FastAPI + ML pipeline)
├── backend/         # Backend service (planned)
├── mobile/          # Mobile app (planned)
└── website/         # Web frontend (planned)
```

## Quick Start

```bash
cd ai-api

# Install dependencies
pip install -e ".[dev]"

# Start infrastructure
docker compose up db redis -d

# Run migrations
alembic upgrade head

# Seed dev data (creates test API key)
python scripts/seed_db.py

# Start the dev server
make dev
```

API docs available at http://localhost:8000/docs after startup.

## Documentation

See [`ai-api/docs/`](ai-api/docs/README.md) for full documentation including architecture, API reference, deployment guide, and training plans.

## Tech Stack

| Category | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| ML Inference | InsightFace, PaddleOCR, Ultralytics YOLOv8, ONNX Runtime |
| Database | PostgreSQL 16 + pgvector |
| Task Queue | Celery + Redis |
| Containerization | Docker Compose (4 services) |

## Development Status

| Phase | Feature | Status |
|---|---|---|
| Phase 1 | Foundation (FastAPI, DB, auth, middleware) | Complete |
| Phase 2 | Blur Detection (Laplacian-based) | Complete |
| Phase 3 | Face Recognition (InsightFace + pgvector) | Complete |
| Phase 4 | Bib Number Recognition (PaddleOCR) | Complete |
| Phase 5 | Async Batch Processing (Celery + Redis) | Complete |
| Phase 6 | C++ Acceleration (pybind11) | Complete |
| Phase 7 | Production Hardening | Pending |
