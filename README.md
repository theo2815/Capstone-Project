# EventAI

A complete ecosystem for marathon photography — combining real-time photo uploading, AI-powered blur detection, face and bib number search, and an online marketplace for runners to find and purchase their event photos.

## Platform

| Product | Technology | Status |
|---------|-----------|--------|
| **Mobile App** | Kotlin (Android first, iOS planned) | Not started |
| **Website** | Next.js (Vercel) | Not started |
| **Desktop App** | Electron | Built |
| **Web/Mobile Backend** | Spring Boot (Java) | Not started |
| **AI Services (ai-api)** | FastAPI + Celery (Python) | Phases 1-6 complete |

## Project Structure

```
Capstone-Project/
├── ai-api/          # AI microservice (FastAPI + ML pipeline) — built
├── backend/         # Web/Mobile backend (Spring Boot) — planned
├── mobile/          # Mobile app (Kotlin) — planned
├── website/         # Web frontend (Next.js) — planned
└── docs/            # Project-level documentation
```

## AI Features (ai-api)

- **Blur Detection** -- Laplacian variance + FFT analysis, and a CNN classifier (YOLOv8n-cls, 98.68% accuracy) for 4-class blur type classification
- **Face Recognition** -- InsightFace (RetinaFace + ArcFace) with pgvector cosine similarity search
- **Bib Number OCR** -- PaddleOCR (PP-OCRv5) with optional YOLOv8 bib region detection
- **Async Batch Processing** -- Celery + Redis for processing 100+ images per request
- **Optional C++ Acceleration** -- pybind11 extensions for performance-critical paths (AVX2)

## Tech Stack

| Category | Technology | Hosting |
|---|---|---|
| Web/Mobile Backend | Spring Boot (Java) | AWS EC2 |
| AI Services | FastAPI + Celery (Python) | AWS EC2 |
| Database | PostgreSQL 16 + pgvector | AWS RDS |
| Object Storage | AWS S3 (CloudFront CDN planned) | AWS |
| Task Queue | Redis | AWS EC2 |
| Website | Next.js | Vercel |
| Mobile | Kotlin | Google Play Store |
| Desktop | Electron | Direct download |

## Quick Start (ai-api)

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

| Document | Description |
|----------|-------------|
| [`docs/project-vision.md`](docs/project-vision.md) | Product vision, problem statement, user journeys, full tech stack |
| [`ai-api/docs/`](ai-api/docs/README.md) | AI service documentation — architecture, API reference, deployment, training plans |
| [`ai-api/docs/integration-architecture.md`](ai-api/docs/integration-architecture.md) | System architecture — responsibility boundaries between ai-api, backends, and desktop |
| [`ai-api/docs/integration-contracts.md`](ai-api/docs/integration-contracts.md) | API contracts — how each backend calls ai-api, with code examples |

## ai-api Development Status

| Phase | Feature | Status |
|---|---|---|
| Phase 1 | Foundation (FastAPI, DB, auth, middleware) | Complete |
| Phase 2 | Blur Detection (Laplacian-based) | Complete |
| Phase 3 | Face Recognition (InsightFace + pgvector) | Complete |
| Phase 4 | Bib Number Recognition (PaddleOCR) | Complete |
| Phase 5 | Async Batch Processing (Celery + Redis) | Complete |
| Phase 6 | C++ Acceleration (pybind11) | Complete |
| Phase 7 | Production Hardening | Pending |
