<div align="center">

<!-- Logo placeholder — replace with the QuickPitik logo when ready -->
<p><em>Logo coming soon</em></p>

# QuickPitik

**A marathon photography ecosystem for Cebu, Philippines.**
Camera → Mobile → Cloud → Marketplace, with AI-powered blur culling on the desktop and face/bib search on the web and mobile.

Live at [quickpitik.com](https://quickpitik.com) · Desktop app: [BatchMyPhotos on the Microsoft Store](https://apps.microsoft.com/detail/9N1KKMV4NX4J)

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
- **Marketplace + checkout** — QR Ph through PayMongo on the website and the Android app, with watermarked previews until purchase.
- **Per-event isolation** — every face search is scoped to one event (one collection per event in AWS Rekognition), so runner data never leaks across races.

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
│  Spring Boot Backend             │   │  ai-api          │
│  users · events · payments       │   │  blur detection  │
└────────────┬─────────────────────┘   └──────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  AWS Rekognition                 │
│  face / bib search               │
└──────────────────────────────────┘
```

**Module boundaries**

- Mobile and website never call `ai-api` directly — face and bib search run in the Spring Boot backend on AWS Rekognition.
- The desktop app is the one exception. It calls `ai-api` with its own restricted key (scopes: `blur:read`, `jobs:read`).
- Each backend owns its own domain data; `ai-api` is stateless about users and events, storing only API keys and blur jobs.

Full integration contracts live in `ai-api/docs/integration-architecture.md` and `ai-api/docs/integration-contracts.md`.

---

## Tech Stack

| Product       | Stack                                                | Hosting           |
|---------------|------------------------------------------------------|-------------------|
| `ai-api/`     | FastAPI · Celery · Python 3.11+ · ONNX Runtime       | Railway           |
| `backend/`    | Spring Boot · Kotlin · PostgreSQL · PayMongo         | Railway           |
| `website/`    | Next.js · React · Tailwind                           | Vercel            |
| `mobile/`     | Kotlin · Jetpack Compose · MVVM (Android 10+)        | APK download      |
| `desktop/`    | Electron · React · Node.js (Windows, separate repo)  | Microsoft Store |
| Database      | PostgreSQL 16 (backend; the blur service has its own)| Railway           |
| Object Storage| Cloudflare R2 (photos)                               | Cloudflare        |
| Face / bib    | AWS Rekognition (one collection per event)           | AWS               |
| Redis         | Blur service API keys and rate limits                | Railway           |

---

## AI Capabilities (ai-api)

- **Blur Detection** — Laplacian variance + FFT analysis, then a person detector (YOLOv8n-seg, ONNX) that measures how sharp each runner is; each photo gets one of four labels. Real-event accuracy is not yet qualified; see `ai-api/docs/ai-system-overview.md`.
- **Face and bib search are not in ai-api** — the Spring Boot backend runs them on AWS Rekognition.
- **Async Batch Processing** — Celery + Redis for processing 100+ images per request; turned off on the live blur service.
- **Optional C++ Acceleration** — pybind11 extensions for performance-critical paths (AVX2).

---

## Monorepo Structure

```
capstone-project/
├── ai-api/        FastAPI + Celery — blur detection for the desktop app
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

- **ai-api**: Python 3.11+, Docker (PostgreSQL, Redis)
- **backend**: JDK 21, PostgreSQL, Gradle
- **website**: Node.js 22, npm
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

| Name                          | Role                       |
|-------------------------------|----------------------------|
| Chan, Theo Cedric             | Lead Developer             |
| Tapales, Christian Kyle       | Developer                  |
| Ycoy, Dillan Marquin          | Developer                  |
| Pureza, Kristine Eunice       | Lead Documents, QA, Tester |
| Sy, Brye Kane L.              | Supporting Documents       |

**Adviser:** Joemarie C. Amparo

---

## Institution

Capstone project, **Cebu Institute of Technology – University (CIT-U)**, Cebu, Philippines.

---

## License

**Proprietary — All Rights Reserved.**

Copyright © 2026 QuickPitik Team. This software and its source code are the proprietary property of the QuickPitik capstone team and Cebu Institute of Technology – University. No part of this project may be copied, modified, distributed, or used in any form without explicit written permission from the authors.
