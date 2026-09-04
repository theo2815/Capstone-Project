<div align="center">

<!-- Logo placeholder — replace with the QuickPitik logo when ready -->
<p><em>Logo coming soon</em></p>

# QuickPitik

**A marathon photography ecosystem for Cebu, Philippines.**
Camera → Mobile → Cloud → Marketplace, with AI-powered blur culling on the desktop and face/bib search on the web and mobile.

</div>

---

## About

Marathon and fun-run photography in the Philippines has two unsolved problems:

- **Photographers** can't upload in real time, manually cull thousands of shots after each event (1–2 hours of work), and have no blur-culling tool tailored for sports photography.
- **Runners** can't find their own photos in a sea of thousands of images and have no local platform to buy them.

QuickPitik addresses both ends of the pipeline. Photographers tether their camera to a mobile device for near real-time upload, use the BatchMyPhotos desktop app to AI-cull blurry shots before publishing, and reach customers directly through the marketplace. Runners search for their photos by selfie or bib number on the website or mobile app, then check out in-app.

**North-star outcomes**

- Photographers: post-event sort time from 1–2 hours → 5–10 seconds.
- Runners: photos appear within minutes of being taken, and are searchable by face or bib.

---

## Key Features

- **Real-time camera tethering** (mobile, Android-first) — capture flows directly from camera to cloud.
- **AI blur culling** (BatchMyPhotos desktop) — photographers remove blurry shots in seconds, not hours.
- **Face search** (web + mobile) — runners upload a selfie and find every photo they appear in for a given event.
- **Bib-number search** — OCR-based search for runners who prefer not to upload a selfie.
- **Marketplace + checkout** — local payment methods (PayMongo) and watermarked previews until purchase.
- **Per-event isolation** — face embeddings are scoped to a single event so runner data never leaks across races.

---

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Mobile App  │   │   Website    │   │  Desktop App │
│  (Kotlin)    │   │  (Next.js)   │   │  (Electron)  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────────────────────────┐   ┌──────────────────┐
│  Spring Boot Backend             │   │  Desktop Backend │
│  users · events · payments       │   │  own DB          │
└────────────┬─────────────────────┘   └────────┬─────────┘
             │                                  │
             ▼                                  ▼
┌──────────────────────────────────────────────────────┐
│  ai-api                                              │
│  · web + mobile use: face / bib                      │
│  · desktop uses:     blur / batch                    │
└──────────────────────────────────────────────────────┘
```

**Module boundaries**

- Mobile and website never call `ai-api` directly — the Spring Boot backend proxies all inference.
- The desktop app is the one exception. It calls `ai-api` with its own restricted key (scopes: `blur:read`, `jobs:read`).
- Each backend owns its own domain data; `ai-api` is stateless about users and events, storing only face embeddings tagged with `api_key_id` + `event_id`.

Full integration contracts live in `ai-api/docs/integration-architecture.md` and `ai-api/docs/integration-contracts.md`.

---

## Tech Stack

| Product       | Stack                                                | Hosting           |
|---------------|------------------------------------------------------|-------------------|
| `ai-api/`     | FastAPI · Celery · Python 3.11+ · pgvector           | AWS EC2           |
| `backend/`    | Spring Boot · Kotlin · PostgreSQL · PayMongo         | AWS EC2           |
| `website/`    | Next.js · React · Tailwind                           | Vercel            |
| `mobile/`     | Kotlin · Jetpack Compose · MVVM (Android)            | Google Play Store |
| `desktop/`    | Electron · React · Node.js (Windows, separate repo)  | Microsoft Store |
| Database      | PostgreSQL 16 + pgvector                             | AWS RDS           |
| Object Storage| AWS S3 (CloudFront CDN planned)                      | AWS               |
| Task Queue    | Redis                                                | AWS EC2           |

---

## AI Capabilities (ai-api)

- **Blur Detection** — Laplacian variance + FFT analysis, and a CNN classifier (YOLOv8n-cls, 98.68% accuracy) for 4-class blur type classification.
- **Face Recognition** — InsightFace (RetinaFace + ArcFace) with pgvector cosine similarity search.
- **Bib Number OCR** — PaddleOCR (PP-OCRv5) with optional YOLOv8 bib region detection.
- **Async Batch Processing** — Celery + Redis for processing 100+ images per request.
- **Optional C++ Acceleration** — pybind11 extensions for performance-critical paths (AVX2).

---

## Monorepo Structure

```
capstone-project/
├── ai-api/        FastAPI + Celery — face / bib / blur inference
├── backend/       Spring Boot — users, events, orders, payments
├── website/       Next.js — public site, runner search, admin
├── mobile/        Kotlin/Compose — photographer + runner apps
└── docs/          Project-wide documentation
```

The desktop app (BatchMyPhotos) lives in its own repository:
[github.com/theo2815/Batch-My-Photos](https://github.com/theo2815/Batch-My-Photos).

---

## Getting Started

Each module has its own setup. Start with the module's `CLAUDE.md` and `docs/` folder.

### Prerequisites

- **ai-api**: Python 3.11+, PostgreSQL with pgvector, Redis
- **backend**: JDK 21, PostgreSQL, Gradle
- **website**: Node.js 20+, npm
- **mobile**: Android Studio, JDK 17, an Android device or emulator

### Quick start — ai-api

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

API docs at http://localhost:8000/docs after startup.

### Quick start — other modules

```bash
# backend (Spring Boot)
cd backend && ./gradlew bootRun

# website (Next.js)
cd website && npm install && npm run dev

# mobile (Android)
# Open the mobile/ folder in Android Studio and run the app configuration.
```

### Environment

All API keys, secrets, and `.env` files are gitignored. Production-like credentials live in `docs/api-keys.md`, which is also gitignored — contact the team for access.

---

## Documentation

| Topic                                | File                                                  |
|--------------------------------------|-------------------------------------------------------|
| Project vision and user journeys     | `docs/project-vision.md`                              |
| Phased implementation roadmap        | `docs/IMPLEMENTATION_PLAN.md`                         |
| ai-api architecture                  | `ai-api/docs/architecture.md`                         |
| Backend ↔ ai-api integration         | `ai-api/docs/integration-architecture.md`             |
| Integration contracts                | `ai-api/docs/integration-contracts.md`                |
| Desktop ↔ ai-api guide               | `docs/desktop-blur-detection-integration-guide.md`    |

Each module also has its own `CLAUDE.md` and `docs/` directory with module-specific rules and references.

---

## Project Status

QuickPitik is in active development. The AI service, Spring Boot backend, website, and mobile app are all implemented and integrated; production hardening and a real-event beta are the remaining arcs. Phase-by-phase history and the live roadmap live in **`docs/IMPLEMENTATION_PLAN.md`** — this README does not restate per-module status, which drifts.

---

## Team

| Name                          | Role               |
|-------------------------------|--------------------|
| Chan, Theo Cedric             | Lead Developer     |
| Tapales, Christian Kyle       | Developer          |
| Ycoy, Dillan Marquin          | Developer          |
| Purez, Kristine Eunice        | Lead Documents     |
| Sy, Brye Kane L.              | Documents          |

**Adviser:** Joemarie C. Amparo

---

## Institution

Capstone project, **Cebu Institute of Technology – University (CIT-U)**, Cebu, Philippines.

---

## License

**Proprietary — All Rights Reserved.**

Copyright © 2026 QuickPitik Team. This software and its source code are the proprietary property of the QuickPitik capstone team and Cebu Institute of Technology – University. No part of this project may be copied, modified, distributed, or used in any form without explicit written permission from the authors.
