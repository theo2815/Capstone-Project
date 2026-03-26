# Integration Architecture — EventAI Platform

**Date:** 2026-03-26
**Status:** Approved design direction
**Resolves:** ARCH-1 (feature-analysis-report.md)

---

## System Overview

EventAI is a **multi-product platform** that serves three client applications through a shared AI microservice:

| Product | Platform | Features Used |
|---------|----------|---------------|
| EventAI Desktop | Electron (already built) | Blur detection + classification |
| EventAI Web | Next.js (Vercel) | Blur + Face search + Bib recognition |
| EventAI Mobile | Kotlin (Android first, iOS planned) | Blur + Face search + Bib recognition |

Each product has its **own backend**. All backends share a **single ai-api instance** for ML inference.

---

## Architecture Diagram

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Desktop App │  │  Website     │  │  Mobile App  │
│  (Electron)  │  │  (Next.js /  │  │  (Kotlin /   │
│              │  │   Vercel)    │  │   Android)   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │                 └────────┬────────┘
       │                          │
┌──────▼───────┐       ┌──────────▼───────────┐
│  Desktop     │       │  Web/Mobile          │
│  Backend     │       │  Backend             │
│              │       │                      │
│  Owns:       │       │  Owns:               │
│  - User auth │       │  - User auth         │
│  - Photo     │       │  - Events            │
│    library   │       │  - Participants      │
│  - Blur      │       │  - Photo galleries   │
│    results   │       │  - Bib-to-person     │
│    storage   │       │    matching          │
│              │       │  - Face galleries     │
│              │       │    per event         │
│              │       │  - Search results    │
│              │       │    storage           │
└──────┬───────┘       └──────────┬───────────┘
       │                          │
       │   ┌──────────────────┐   │
       └──►│     ai-api       │◄──┘
            │  (shared ML      │
            │   microservice)  │
            │                  │
            │  Owns:           │
            │  - ML models     │
            │  - Face          │
            │    embeddings    │
            │  - Batch job     │
            │    queue         │
            │  - Raw inference │
            │    results       │
            └──────────────────┘
```

---

## Responsibility Boundary

This is the single source of truth for what each layer owns. If a concern is not listed under a layer, that layer must NOT implement it.

### ai-api (ML microservice) — owns inference

| Responsibility | Details |
|---------------|---------|
| Blur detection | Laplacian variance + FFT analysis |
| Blur classification | YOLOv8n-cls 4-class CNN (sharp, defocused_blurred, defocused_object_portrait, motion_blurred) |
| Face detection | InsightFace RetinaFace — bounding boxes + landmarks |
| Face embedding | InsightFace ArcFace — 512-dim vectors |
| Face storage | Store embeddings in pgvector, scoped by `api_key_id` + `event_id` |
| Face search | Cosine similarity search against stored embeddings |
| Bib OCR | PaddleOCR — extract text from bib regions |
| Bib detection | YOLOv8 — locate bib regions in image (when model available) |
| Batch processing | Celery workers for async multi-image jobs |
| Webhooks | Notify caller when batch job completes |

**ai-api does NOT own:**
- User accounts, sessions, or login
- Event or participant data
- Bib-to-participant matching
- Confidence threshold business rules
- Photo storage (processes in memory, discards after)
- Client-facing API design (response shaping for UI)

### Desktop Backend — owns photo library + blur workflow

| Responsibility | Details |
|---------------|---------|
| User authentication | Desktop app login (local or cloud accounts) |
| Photo library | Import, organize, tag photos locally or in cloud storage |
| Blur filtering | Call ai-api for blur detection/classification, store results, let user filter |
| Result storage | Persist blur scores and classifications per photo |
| UI-ready responses | Shape ai-api raw results into what the desktop UI needs |
| Offline fallback | (Future) Optional local ONNX inference if ai-api unreachable |

**Desktop Backend does NOT own:**
- Events, participants, or bib numbers
- Face recognition or search
- ML model management

### Web/Mobile Backend — owns events + full search workflow

| Responsibility | Details |
|---------------|---------|
| User authentication | Web/mobile login, roles (admin, photographer, runner/participant) |
| Event management | Admin creates public event galleries, configures settings per event |
| Participant management | Import participant lists (name, bib number, category) |
| Photo upload + storage | Receive photos from photographers, store in cloud |
| Blur quality gate | Call ai-api blur detect/classify, reject or flag blurry photos |
| Face enrollment | On event setup, call ai-api to enroll participant face photos, **always pass `event_id`** |
| Face search | On photo upload, call ai-api face search to auto-tag participants in photos, **always pass `event_id`** |
| Bib recognition | On photo upload, call ai-api bib OCR, then match bib text to participant list |
| Confidence filtering | Apply per-event thresholds before showing results to users |
| Result aggregation | Combine face match + bib match + manual tags into final photo-participant mapping |
| Client API | REST/GraphQL API that web and mobile apps consume |

**Web/Mobile Backend does NOT own:**
- ML model inference (delegates to ai-api)
- Face embedding storage (ai-api handles via pgvector)
- Raw image processing (ai-api handles)

---

## API Key Strategy

Each backend gets its own API key with appropriate scopes. ai-api isolates data by `api_key_id`.

| Backend | API Key Scope | Rate Tier |
|---------|--------------|-----------|
| Desktop Backend | `blur:detect`, `blur:classify` | Internal (1000 req/min) |
| Web/Mobile Backend | `blur:*`, `faces:*`, `bibs:*` | Internal (1000 req/min) |

- **Never expose ai-api keys to client apps.** Clients talk to their own backend only.
- Each API key creates an isolated data space in ai-api (separate face embeddings, separate job history).
- If multiple environments exist (staging, production), use separate API keys per environment.

---

## Event Isolation (resolves FA-1)

The most critical architecture requirement is preventing cross-event data leakage in face search.

### How it works

1. Web/Mobile Backend creates an event (e.g., "Marathon 2026") and stores it in its own DB.
2. When enrolling faces, backend passes `event_id` as a form field to ai-api's `/faces/enroll`.
3. ai-api stores `event_id` alongside the face embedding.
4. When searching faces, backend passes `event_id` as a query parameter to ai-api's `/faces/search`.
5. ai-api filters search results to only match faces within that event.

```
Web/Mobile Backend                          ai-api
       │                                      │
       │  POST /faces/enroll                  │
       │  form: file, person_name, event_id   │
       │─────────────────────────────────────►│
       │                                      │  Store embedding with
       │                                      │  api_key_id + event_id
       │                                      │
       │  POST /faces/search?event_id=abc     │
       │  form: file                          │
       │─────────────────────────────────────►│
       │                                      │  Search WHERE api_key_id = X
       │                                      │  AND event_id = 'abc'
       │  { matches: [...] }                  │
       │◄─────────────────────────────────────│
```

Desktop Backend does not use face search, so event isolation does not apply to it.

---

## Bib Matching Flow (resolves BIB-1)

ai-api performs OCR only. The Web/Mobile Backend matches bib text to participants.

```
Web/Mobile Backend                          ai-api
       │                                      │
       │  POST /bibs/recognize                │
       │  form: file                          │
       │─────────────────────────────────────►│
       │                                      │  Run YOLO detection + OCR
       │  { bib_number: "1023",               │
       │    confidence: 0.91,                 │
       │    bbox: {...} }                     │
       │◄─────────────────────────────────────│
       │                                      │
       │  Backend logic:                      │
       │  1. Check confidence >= event threshold (e.g., 0.7)
       │  2. SELECT * FROM participants WHERE bib_number = '1023' AND event_id = X
       │  3. If match found, link photo to participant
       │  4. If no match, flag for manual review
```

---

## Confidence Threshold Strategy (resolves BIB-2)

ai-api always returns raw confidence scores. Each backend decides what to do with them.

| Feature | ai-api returns | Backend applies |
|---------|---------------|-----------------|
| Blur detect | `confidence: 0.85, is_blurry: false` | Desktop: show to user as quality score. Web: auto-reject if `is_blurry = true`. |
| Blur classify | `predicted_class: "motion_blurred", confidence: 0.72` | Desktop: tag photo with blur type. Web: reject if blur confidence > event threshold. |
| Face search | `similarity: 0.65` | Web: show match only if similarity >= event's face threshold (e.g., 0.6) |
| Bib OCR | `bib_number: "1023", confidence: 0.45` | Web: discard if confidence < event's bib threshold (e.g., 0.7) |

This means:
- ai-api never discards results based on business-level thresholds
- Each backend configures thresholds per event or per user preference
- The same ai-api response can be interpreted differently by desktop vs web

---

## Network Topology

### Production (AWS)

```
                    ┌──────────────────┐
                    │  Vercel          │
                    │  └── Website     │
                    │     (Next.js)    │
                    └────────┬─────────┘
                             │ calls
                             ▼
┌─────────────────────────────────────────────────────┐
│  AWS EC2                                             │
│  ┌─────────────────────────────────────────────┐     │
│  │  Spring Boot Backend  (public, port 8080)   │     │
│  └──────────────────┬──────────────────────────┘     │
│                     │ internal                       │
│  ┌──────────────────▼──────────────────────────┐     │
│  │  ai-api (FastAPI)  (private, port 8000)     │     │
│  ├─────────────────────────────────────────────┤     │
│  │  Celery Worker     (private)                │     │
│  ├─────────────────────────────────────────────┤     │
│  │  Redis             (private, port 6379)     │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────┐
│  AWS RDS                                             │
│  └── PostgreSQL 16 + pgvector (private)              │
│      (→ Qdrant migration planned if latency issue)   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  AWS S3 (+ CloudFront CDN planned)                   │
│  └── Photo storage                                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Desktop App (Electron — on photographer's machine)  │
│  └── Desktop Backend → calls ai-api over HTTPS       │
│  └── Desktop DB (local or cloud)                     │
└─────────────────────────────────────────────────────┘
```

- ai-api is **private within EC2** — only the Spring Boot backend and Desktop Backend can reach it.
- Spring Boot Backend is the public API for website and mobile app.
- Desktop Backend connects to ai-api over HTTPS with its own API key.
- Mobile app and website **never** talk to ai-api directly.

### Development

```
localhost:3000  ← Desktop Backend
localhost:4000  ← Web/Mobile Backend (Spring Boot)
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
| Face embeddings | ai-api | PostgreSQL + pgvector (AWS RDS) | ai-api |
| Blur scores | ai-api (computed) | Backend DBs (persisted) | Backends |
| Bib OCR text | ai-api (computed) | Backend DBs (persisted) | Web/Mobile Backend |
| Batch jobs | ai-api | ai-api PostgreSQL + Redis | ai-api → webhook → Backends |
| API keys | ai-api admin | ai-api PostgreSQL | Backends (as credentials) |

---

## What Needs to Change in ai-api

Based on this architecture, the following changes are needed in ai-api before production:

| Change | Priority | Report ID | Description |
|--------|----------|-----------|-------------|
| Add `event_id` to face operations | CRITICAL | FA-1 | Add optional `event_id` param to enroll/search/batch-search. Add column to Person model. Filter by it when present. |
| Return raw confidence always | HIGH | BIB-2 | Already done — just ensure no server-side filtering is added. Document that backends must filter. |
| Fix dead threshold param in blur detect | MEDIUM | BLUR-3 | Either pass it through to detector or remove the query parameter. |
| Add duplicate embedding guard | MEDIUM | FA-6 | Check `source_image_hash` + `person_id` before storing. |
| Add list-persons endpoint | MEDIUM | FA-7 | `GET /faces/persons` scoped by api_key_id (+ event_id). |

Changes that are NOT needed in ai-api (backend's job):
- Bib-to-participant matching (BIB-1) — backend handles
- Per-event confidence thresholds (BIB-2) — backend applies
- Event/participant management — backend owns
