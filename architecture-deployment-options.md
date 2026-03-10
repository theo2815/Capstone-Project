# Event Photo Platform — Architecture Recap

## Project Overview

A platform for running event photographers to upload photos and for runners to find/purchase their photos using AI-powered search (face recognition + bib number).

### Platform Components

- **Mobile Application** (Android first)
- **Website** (Web App)
- **Backend API** (Spring Boot)
- **AI API Service** (FastAPI — already built)
- **Desktop App** (calls AI API directly)

---

## Tech Stack Decisions

| Component | Technology | Hosting |
|-----------|-----------|---------|
| **Website** | Next.js | Vercel (free) |
| **Backend API** | Spring Boot (Java) | VPS in Singapore |
| **AI API** | FastAPI + Celery (Python) — already built | Same VPS |
| **Mobile App** | Android (Kotlin) | Google Play Store (Internal Testing) |
| **Desktop App** | — | Calls AI API directly |
| **Database** | PostgreSQL 16 + pgvector | Same VPS (Docker) |
| **Cache / Queue** | Redis | Same VPS (Docker) |
| **Photo Storage** | Cloudflare R2 | Cloudflare (free tier — 10GB, zero egress) |
| **Reverse Proxy** | Nginx | Same VPS |
| **SSL** | Let's Encrypt | Free, auto-renew |

---

## AI Models (Status)

| Model | Status | Purpose |
|-------|--------|---------|
| Blur Detection (Laplacian + CNN) | Completed | Filter out blurry photos before storage |
| Face Recognition (InsightFace) | Completed | Runners search photos by selfie |
| Bib Number OCR (PaddleOCR) | Completed | Runners search photos by bib number |
| Face + Bib Detector (YOLOv8) | Training paused | Improved detection accuracy |

---

## Architecture Style

- **Modular monolith** for the backend — single Spring Boot service, not microservices
- **AI API as a separate service** — but deployed on the **same VPS** in Docker
- **Combined deployment** — all services run in Docker Compose on one VPS
- Can split to separate servers later if needed (just a config change)

---

## Deployment

### Hosting

| What | Where | Cost |
|------|-------|------|
| VPS | **Contabo Cloud VPS M — Singapore** (6 vCPU, 16GB RAM, 400GB SSD) | ~$13/mo |
| Website | **Vercel** | Free |
| Photo storage | **Cloudflare R2** | Free (up to 10GB, zero egress fees) |
| Domain | Any registrar | ~$10/year |
| SSL | Let's Encrypt | Free |
| Play Store | Google Play Console | $25 one-time |
| **Total recurring** | | **~$13/mo** |

### Why Contabo Singapore (not Hetzner EU)

- Hetzner only has EU data centers → ~200-300ms latency from Philippines
- Contabo Singapore → ~30-50ms latency from Philippines
- 16GB RAM at ~$13/mo — best value for Asia region

### VPS Memory Budget (16GB)

| Component | RAM Usage |
|-----------|-----------|
| AI Models (InsightFace, PaddleOCR, YOLOv8, Blur CNN) | ~2.0 GB |
| FastAPI AI API | ~300 MB |
| Celery Worker (loads same models) | ~2.0 GB |
| Spring Boot Backend | ~512 MB |
| PostgreSQL | ~512 MB |
| Redis | ~128 MB |
| Nginx | ~32 MB |
| OS overhead | ~512 MB |
| **TOTAL** | **~6.0 GB** |
| **Headroom** | **~10 GB free** |

---

## URL Routing

### DNS Setup

```
yourdomain.com         CNAME → cname.vercel-dns.com    (website on Vercel)
api.yourdomain.com     A     → <VPS IP>                (backend + AI API)
```

### Nginx Routes (on VPS)

| URL | Routes To |
|-----|-----------|
| `yourdomain.com` | Website (Vercel) |
| `api.yourdomain.com/api/*` | Spring Boot backend (:8080) |
| `api.yourdomain.com/ai/*` | FastAPI AI API (:8001) |

---

## Who Calls What

| Client | Calls | URL |
|--------|-------|-----|
| Website (Vercel) | Backend only | `api.yourdomain.com/api/*` |
| Mobile App (Android) | Backend only | `api.yourdomain.com/api/*` |
| Desktop App | AI API directly | `api.yourdomain.com/ai/*` |
| Spring Boot (internally) | AI API via Docker network | `http://ai-api:8001` (no public internet hop) |

---

## Database Design

- **Single PostgreSQL instance**, shared by Backend and AI API
- **pgvector extension** for face embedding similarity search (HNSW index, sub-10ms for 1M embeddings)
- **GIN index** on bib number arrays for instant bib search
- Backend uses Spring Data JPA / Flyway migrations
- AI API uses SQLAlchemy + Alembic migrations (already set up)

### Why PostgreSQL (not MongoDB)

- Data is relational (events → photographers → photos → orders → payments)
- pgvector provides native face embedding search — no separate vector DB needed
- AI API already uses PostgreSQL + pgvector
- ACID transactions for payment flows

---

## Photo Storage Strategy

### Cloudflare R2

- S3-compatible object storage
- **Zero egress fees** — critical for a photo platform with many image views
- Free tier: 10GB storage, 10M Class B requests/mo
- Upload via **presigned URLs** — clients upload directly to R2, bypassing backend

### Storage Layout

```
bucket: event-photos/
└── events/{event_id}/
    ├── raw/{photo_id}.jpg          # Originals (temporary)
    ├── published/{photo_id}.jpg    # AI-verified sharp images
    ├── thumbs/sm/{photo_id}.jpg    # 150px gallery grid
    ├── thumbs/md/{photo_id}.jpg    # 600px preview (watermarked)
    └── originals/{photo_id}.jpg    # Full-res for purchased downloads
```

### Upload Flow (Presigned URLs)

1. Client requests presigned URL from Spring Boot backend
2. Client uploads image directly to Cloudflare R2 (skips backend)
3. Client confirms upload → backend queues AI processing
4. AI pipeline: **Blur filter → Face embedding + Bib OCR → Thumbnails**
5. Blurry photos rejected, sharp photos published

---

## Docker Compose Services (on VPS)

| Service | Image / Build | Port |
|---------|--------------|------|
| `postgres` | pgvector/pgvector:pg16 | 5432 (internal) |
| `redis` | redis:7-alpine | 6379 (internal) |
| `backend` | ./backend (Spring Boot) | 8080 (internal) |
| `ai-api` | ./ai-api (FastAPI) | 8001 (internal) |
| `celery-worker` | ./ai-api (Celery) | — |
| `nginx` | nginx:alpine | 80, 443 (public) |

All ports are internal except Nginx. Nginx handles SSL termination and routes traffic.

---

## Mobile App Deployment

- **Google Play Console** — $25 one-time registration fee
- Use **Internal Testing track** for capstone demo (no review needed, instant access via link)
- Build signed AAB in Android Studio
- Release build points to `https://api.yourdomain.com/api/`
- Debug build points to local dev server

---

## Scaling Path (If Needed Later)

Current setup handles capstone scale easily. If real production traffic arrives:

1. **Split AI API to separate VPS** — just change `AI_API_URL` config from `http://ai-api:8001` to `https://ai.yourdomain.com`
2. **Add GPU server** for faster inference (Vast.ai, RunPod)
3. **Add CDN** (Cloudflare) in front of Nginx for caching
4. **PostgreSQL read replica** if search queries spike
5. **Scale Celery workers** independently for batch processing

No code changes needed for any of these — just infrastructure config.

---

## Key Design Decisions Summary

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Backend architecture | Modular monolith (Spring Boot) | Small team, capstone scope, avoid microservice overhead |
| AI service | Separate FastAPI service, same VPS | Already built, serves desktop app directly |
| Database | PostgreSQL + pgvector | Relational data, vector search, already in use |
| Photo storage | Cloudflare R2 | Zero egress, S3-compatible, free tier |
| Upload method | Presigned URLs (direct to R2) | Backend never touches image bytes, saves bandwidth |
| VPS region | Singapore | Low latency from Philippines (~30-50ms) |
| VPS provider | Contabo | Best value for 16GB RAM in Asia (~$13/mo) |
| Website hosting | Vercel | Free, global CDN, auto-deploy from Git |
| Combined vs separated | Combined (single VPS) | Cost-efficient, simple, can split later |
