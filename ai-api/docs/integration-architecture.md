# Integration Architecture — QuickPitik Platform

**Date:** 2026-04-23
**Status:** Approved design direction · **Updated 2026-10-16:** face and bib retired from ai-api

---

## System Overview

QuickPitik is a **multi-product platform** with three client applications. ai-api serves only one of them:

| Product | Platform | Features Used |
|---------|----------|---------------|
| QuickPitik Desktop | Electron (already built) | Blur detection + classification (ai-api) |
| QuickPitik Web | Next.js (Vercel) | Face search + Bib recognition through the backend, not ai-api (no blur — rule 6) |
| QuickPitik Mobile | Kotlin (Android first, iOS planned) | Face search + Bib recognition through the backend, not ai-api (no blur — rule 6) |

**Retired 2026-10-16:** the backend uses AWS Rekognition (`AI_PROVIDER=rekognition`, set on staging and production; the backend's own default is still `ai-api`, whose face/bib calls now get 404s); ai-api serves blur only, to the desktop app. The ai-api face and bib routes, workers and models were removed. The BatchMyPhotos Electron main process calls ai-api directly for blur classification.

---

## Architecture Diagram

```
BatchMyPhotos Electron main ── HTTPS + restricted key ──► ai-api (blur)
Website / Mobile ──► Spring Boot backend ──► AWS Rekognition (face / bib)
```

---

## Responsibility Boundary

This is the single source of truth for what each layer owns. If a concern is not listed under a layer, that layer must NOT implement it.

### ai-api (ML microservice) — owns inference

| Responsibility | Details |
|---------------|---------|
| Blur detection | Laplacian variance + FFT analysis |
| Blur classification | Two-stage blur classifier (whole-frame rule, then person detector + per-person sharpness + logistic rule); four classes reported (sharp, defocused_blurred, defocused_object_portrait, motion_blurred) |
| Batch processing | Celery workers for async multi-image blur jobs (`/batch`, `/mega`) |
| Webhooks | Notify caller when batch job completes — fan-out scoped to the owning `api_key_id`; payload is `{job_id, result_count}`, results fetched via `GET /jobs/{id}` (webhook-or-poll). |

**ai-api does NOT own:**
- User accounts, sessions, or login
- Event or participant data
- Bib-to-participant matching
- Confidence threshold business rules
- Photo storage (single and stream requests are processed in memory; batch/mega inputs are staged only until `BLOB_RETENTION_SECONDS`, default 24 h, after the job ends)
- Client-facing API design (response shaping for UI)

### BatchMyPhotos desktop — owns local blur workflow

| Responsibility | Details |
|---------------|---------|
| User authentication | Existing BatchMyPhotos sign-in; feedback uses its Supabase session |
| Photo library | Local file groups, preview, and ordinary batch operations |
| Blur suggestions | On explicit Start Analysis, main prepares JPEGs and calls ai-api classify/stream directly |
| Result handling | Map raw probabilities to advisory categories and sensitivity; keep one-folder results in memory |
| Feedback | Separate opt-in submission of one prepared JPEG through Supabase; not part of inference |

**BatchMyPhotos desktop does NOT own:**
- Events, participants, or bib numbers
- Face recognition or search
- ML model management

### Web/Mobile Backend — no longer an ai-api caller

Retired 2026-10-16: the backend uses AWS Rekognition (`AI_PROVIDER=rekognition`, set on staging and production; its code default is still `ai-api`) for face and bib work; ai-api serves blur only, to the desktop app. The backend's events, participants, thresholds and photo storage are documented in the backend module, not here.

---

## API Key Strategy

Each caller gets its own API key with appropriate scopes. ai-api isolates data by `api_key_id`.

| Caller | API Key Scopes | Rate Tier |
|---------|--------------|-----------|
| Invited desktop beta | `blur:read`, `jobs:read` | Per-tester staging tier recorded in private `docs/api-keys.md` |

- **Never expose ai-api keys to website or mobile clients.** The desktop exception keeps its restricted key encrypted in Electron main and sends only presence to the renderer.
- Each API key creates an isolated data space in ai-api (separate job history and webhook subscriptions).
- If multiple environments exist (staging, production), use separate API keys per environment.

---

## Confidence Threshold Strategy (resolves BIB-2)

ai-api always returns raw confidence scores. The caller decides what to do with them.

| Feature | ai-api returns | Backend applies |
|---------|---------------|-----------------|
| Blur detect | `confidence: 0.85, is_blurry: false` | Available to API clients, but the invited desktop beta does not call this route. Web/mobile do not call blur (rule 6). |
| Blur classify | `predicted_class: "motion_blurred", confidence: 0.72` | Desktop maps raw probabilities to advisory suggestions; it never automatically culls or moves a photo. Web/mobile do not call blur (rule 6). |

This means:
- ai-api never discards results based on business-level thresholds
- The desktop app sets its own sensitivity on the raw probabilities

---

## Network Topology

### Deployment (dedicated blur service)

```
┌─────────────────────────────────────────────────────┐
│  BatchMyPhotos (Electron — photographer's machine)   │
│  └── Main process → blur ai-api over HTTPS            │
│  └── Restricted key encrypted locally                │
└─────────────────────────────────────────────────────┘
```

- The blur service is its own deployment (`.railway/railway.ts`, service `ai-api-blur`), with PostgreSQL and Redis for keys, jobs and rate limiting.
- BatchMyPhotos main connects directly to ai-api with its own restricted key.
- Mobile app and website **never** talk to ai-api; the Spring Boot backend does not either while `AI_PROVIDER=rekognition`.

### Development

```
Electron main   ← BatchMyPhotos blur client
localhost:8000  ← ai-api (FastAPI)
localhost:5432  ← PostgreSQL
localhost:6379  ← Redis
```

---

## Data Flow Summary

| Data | Created by | Stored in | Used by |
|------|-----------|-----------|---------|
| User accounts | Backends | Backend DBs | Backends |
| Events | Web/Mobile Backend | Web/Mobile DB (AWS RDS) | Web/Mobile Backend |
| Participants | Web/Mobile Backend | Web/Mobile DB (AWS RDS) | Web/Mobile Backend |
| Photos (files) | Backends | AWS S3 | Backends |
| Face embeddings | ai-api, until 2026-10-16 (retired) | ai-api PostgreSQL (`persons`, `face_embeddings`; rows remain) | nothing |
| Blur probabilities | ai-api (computed) | Desktop one-folder memory cache | BatchMyPhotos advisory UI |
| Batch jobs | ai-api | ai-api PostgreSQL + Redis | ai-api → webhook or poll → caller |
| API keys | ai-api admin | ai-api PostgreSQL | Desktop app (as credentials) |

---

## What ai-api does NOT own

(Backend's job — listed here because these come up in design reviews.)

- Confidence thresholds — the caller applies its own threshold to the raw scores ai-api returns
- Event / participant management — the Web/Mobile Backend owns these and no longer calls ai-api
