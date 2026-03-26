# EventAI — Project Vision

**Date:** 2026-03-26
**Type:** Capstone Project / Startup
**Location:** Cebu, Philippines
**Target Market:** Marathon and running event photography

---

## Problem Statement

### For Photographers

Marathon photographers in Cebu face a painful post-event workflow:

1. **No real-time upload.** Photographers capture thousands of photos during the event but can only upload after it ends. They must manually transfer images from camera SD card to computer first.

2. **Manual organization is slow.** After transfer, photographers must sort thousands of images into folders (typically 500-1000 photos per folder, depending on upload platform requirements). This takes **1-2 hours or more** of tedious work.

3. **No automatic quality culling.** There is no system to detect blurred or unusable images. Photographers must review each photo manually, wasting time on images that should be discarded.

### For Runners

Event participants also struggle:

1. **Finding your photos is a needle-in-a-haystack problem.** After the event, runners must manually browse through thousands of photos to find pictures of themselves.

2. **No dedicated platform.** There is no local platform where runners can easily search for and purchase their specific marathon photos.

---

## Competitor Analysis

### FindMyShots

| Strength | Weakness |
|----------|----------|
| Platform for runners to search and purchase marathon photos | No real-time photo upload during events |
| Face recognition and bib number search | Photographer workflow not addressed |
| Established marketplace model | |

### Zno Instant

| Strength | Weakness |
|----------|----------|
| Camera-to-cloud solution (camera → mobile → cloud) | No marathon-specific features |
| Real-time upload during events | No face or bib number search |
| Quick photo display and sales | Not built for runners finding their own photos |

### Our Advantage

Neither competitor offers the complete pipeline. FindMyShots has the search but not the upload. Zno Instant has the upload but not the search. **EventAI combines both into one platform.**

---

## Proposed Solution

A **Camera → Mobile → Cloud → Marketplace** system that serves both photographers and runners across three platforms.

```
                    PHOTOGRAPHER WORKFLOW
                    ═══════════════════
    Camera ──► Mobile App ──► Cloud Storage ──► AI Processing
                  (real-time      (AWS S3)       (blur filter,
                   upload)                        face + bib)
                                                      │
                                                      ▼
                    RUNNER EXPERIENCE                Marketplace
                    ════════════════                (purchase +
    Runner ──► Mobile App / Website ──► Search ──► download)
                  (face selfie or        (AI-powered
                   bib number)            matching)

                    PHOTOGRAPHER TOOLS
                    ══════════════════
    SD Card ──► Desktop App ──► Blur Detection + Auto-Sort
                  (own backend,       (5-10 seconds vs
                   own database)       1-2 hours manual)
```

---

## Platform Details

### 1. Mobile Application (Kotlin — Android first, iOS planned)

**For Photographers:**
- Connect to camera (WiFi/USB tethering)
- Every photo taken auto-uploads to cloud storage in real-time during the event
- No more waiting until after the event to start uploading
- View upload progress and manage event albums

**For Runners:**
- Create account and log in
- Browse events (e.g., "Cebu Marathon 2026")
- Search for your photos using:
  - **Face recognition** — upload a selfie, find all photos with your face
  - **Bib number search** — enter your bib number, find all photos showing it
- Receive push notifications when your photos are available
- Preview (watermarked) and purchase photos
- Download full-resolution purchased photos

**Key advantage:** Runners can see their photos almost immediately after they are taken, not hours later.

### 2. Website (Next.js on Vercel)

**Purpose:** Alternative platform for users who prefer not to install an app.

**For Photographers:**
- Upload event photos through the website (post-event alternative to mobile real-time upload)
- Manage event galleries and view sales

**For Runners:**
- Same search experience as mobile app (face recognition + bib number)
- Preview, purchase, and download photos
- Browse event galleries

**Why it matters:** Accessibility. Not every runner wants to install an app for a single event. The website ensures the platform reaches all users.

### 3. Desktop Application (Electron — already built)

**Purpose:** Professional tool for photographers handling large photo volumes (5,000-15,000 images per event).

**Architecture:** The desktop app has its **own backend and database**, separate from the web/mobile backend. It communicates with ai-api for ML inference.

**Features:**
- **User accounts** — Photographer login and profile management
- **Automatic blur detection** — AI identifies and separates blurred/unusable images
- **Automatic batch sorting** — Organize photos into folders (e.g., 500 per folder) for upload
- **Batch processing** — Process thousands of images without manual review

**Impact:** A task that normally takes photographers **1-2 hours** (manual review and sorting) is completed in **5-10 seconds**.

**Scope:** The desktop app focuses on blur detection and photo organization. It does not handle face search, bib recognition, or marketplace features — those live on the website and mobile app.

---

## User Journeys

### Journey 1: Photographer During Event (Mobile)

```
1. Photographer arrives at marathon venue
2. Opens EventAI mobile app → Connects camera
3. Race starts → Photographer shoots
4. Every photo auto-uploads to cloud in real-time
5. ai-api runs blur detection on each upload
6. Blurry photos flagged, sharp photos published
7. Runners start seeing their photos within minutes
8. Race ends → All photos already uploaded and processed
9. Photographer goes home (no SD card transfer needed)
```

**Time saved:** Eliminates 1-2 hours of post-event upload + organization.

### Journey 2: Photographer After Event (Desktop)

```
1. Photographer returns home with SD card (if not using mobile upload)
2. Opens EventAI desktop app
3. Imports 10,000 photos from SD card
4. Desktop app auto-detects and separates 800 blurry images
5. Desktop app sorts remaining 9,200 photos into 19 folders of 500
6. Photographer uploads sorted folders to EventAI website
7. Total time: ~5-10 minutes (vs 1-2 hours manual)
```

### Journey 3: Runner Finding Photos (Mobile or Web)

```
1. Runner finishes marathon
2. Opens EventAI app or website
3. Searches for "Cebu Marathon 2026"
4. Taps "Find My Photos"
5. Option A: Takes a selfie → face recognition finds all matching photos
6. Option B: Enters bib number "1023" → bib search finds all photos showing #1023
7. Browses results (watermarked previews)
8. Purchases favorite photos
9. Downloads full-resolution images
```

### Journey 4: Admin Creates Public Event Gallery

```
1. Admin logs in to EventAI admin panel
2. Creates a new event (e.g., "Cebu Marathon 2026") as a public gallery
3. Configures event settings (photo pricing, blur threshold, etc.)
4. Assigns photographers to the event
5. Event goes live → Photographers can start uploading
6. Runners browse the public gallery to find and purchase their photos
```

---

## Feature Matrix

| Feature | Mobile (Photographer) | Mobile (Runner) | Website | Desktop |
|---------|----------------------|-----------------|---------|---------|
| Camera-to-cloud upload | Yes | — | — | — |
| Manual photo upload | — | — | Yes | — |
| Face photo search | — | Yes | Yes | — |
| Bib number search | — | Yes | Yes | — |
| Push notifications | — | Yes | — | — |
| Photo preview (watermarked) | — | Yes | Yes | — |
| Photo purchase + download | — | Yes | Yes | — |
| Blur detection (auto) | Automatic on upload | — | Automatic on upload | Yes (batch) |
| Blur classification | — | — | — | Yes (batch) |
| Batch auto-sort into folders | — | — | — | Yes |
| Event management | — | — | Yes | — |
| Photographer dashboard | — | — | Yes | — |
| Sales analytics | — | — | Yes | — |

---

## Revenue Model

| Revenue Stream | Description |
|---------------|-------------|
| **Photo sales** | Runners purchase individual photos or photo packs. Platform takes a percentage per sale. |
| **Subscription (photographers)** | Monthly plan for photographers: unlimited events, priority processing, desktop app access. |
| **Event packages** | Event organizers pay for a package: AI processing for N photos, event page, participant enrollment. |

---

## Tech Stack

| Component | Technology | Hosting |
|-----------|-----------|---------|
| **Mobile App** | Kotlin (Android first, iOS planned) | Google Play Store |
| **Website** | Next.js | Vercel |
| **Desktop App** | Electron (already built) | Direct download |
| **Web/Mobile Backend** | Spring Boot (Java) | AWS EC2 |
| **Desktop Backend** | (own backend, own database) | TBD |
| **AI Services (ai-api)** | FastAPI + Celery (Python) — already built | AWS EC2 |
| **Database** | PostgreSQL + pgvector | AWS RDS |
| **Object Storage** | AWS S3 | AWS (CloudFront CDN planned if cost allows) |
| **Cache / Queue** | Redis | AWS EC2 (Docker) |

### Vector Search Strategy

Face embeddings are stored in PostgreSQL with pgvector (AWS RDS). If vector search latency becomes an issue at scale, the plan is to migrate to **Qdrant** as a dedicated vector database.

### CDN Strategy

Photo delivery currently goes through AWS S3 directly. **CloudFront CDN** will be added when the cost/benefit makes sense — specifically if S3 egress costs exceed budget limits from high photo view traffic.

---

## How ai-api Fits In

The ai-api is the **AI engine** that powers the platform's core features. It is an internal service — never exposed directly to end users.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Mobile App  │  │   Website    │  │  Desktop App │
│  (Kotlin)    │  │  (Next.js)   │  │  (Electron)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │                 │                 │
       ▼                 ▼                 ▼
┌────────────────────────────┐  ┌──────────────────┐
│  Web/Mobile Backend        │  │  Desktop Backend  │
│  (Spring Boot — AWS EC2)   │  │  (own backend,    │
│                            │  │   own database)   │
│  Events, Users, Payments,  │  │                   │
│  Participant Matching,     │  │  User accounts,   │
│  Photo Marketplace         │  │  Blur results,    │
└────────────┬───────────────┘  │  Photo sorting    │
             │                  └────────┬──────────┘
             │                           │
             ▼                           ▼
┌──────────────────────────────────────────────────┐
│  ai-api (FastAPI + Celery — AWS EC2)             │
│  ════════════════════════════════════             │
│  Blur Detection + Classification                 │
│  Face Recognition + Search (pgvector)            │
│  Bib Number OCR                                  │
│  Batch Processing                                │
├──────────────────────────────────────────────────┤
│  PostgreSQL + pgvector (AWS RDS)                 │
│  Redis (Celery broker + cache)                   │
│  AWS S3 (photo storage)                          │
└──────────────────────────────────────────────────┘
```

| ai-api Feature | Used By | For What |
|----------------|---------|----------|
| Blur detect/classify | Desktop Backend, Web/Mobile Backend | Quality gate — reject blurry photos |
| Face enroll | Web/Mobile Backend | Register participant faces per event |
| Face search | Web/Mobile Backend | Find participants in uploaded photos |
| Bib OCR | Web/Mobile Backend | Read bib numbers from race photos |
| Batch processing | Web/Mobile Backend | Process bulk uploads asynchronously |

See `ai-api/docs/integration-architecture.md` and `ai-api/docs/integration-contracts.md` for the full technical contract between ai-api and the backends.

---

## Deployment

```
┌─────────────────────────────────────────────────────┐
│  Vercel (Free)                                       │
│  └── Website (Next.js)                               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  AWS EC2                                             │
│  ├── Spring Boot Backend                             │
│  ├── ai-api (FastAPI)                                │
│  ├── Celery Worker                                   │
│  └── Redis                                           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  AWS RDS                                             │
│  └── PostgreSQL 16 + pgvector                        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  AWS S3 (+ CloudFront CDN planned)                   │
│  └── Photo storage (originals, thumbnails,           │
│      watermarked previews)                           │
└─────────────────────────────────────────────────────┘
```

### Scaling Path

| When | Action |
|------|--------|
| Vector search gets slow | Migrate face embeddings from pgvector (RDS) to Qdrant |
| S3 egress costs too high | Add CloudFront CDN in front of S3 |
| EC2 CPU bottleneck | Scale ai-api to separate EC2 instance or add GPU instance |
| Celery queue backs up | Add more Celery workers (horizontal scaling) |
| Database reads spike | Add RDS read replica |

---

## Current Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| **ai-api** | Phases 1-6 complete | Blur, face, bib, batch processing, C++ acceleration all built. Production hardening pending. |
| **Desktop App** | Built | Electron app for photographers. Own backend + database. |
| **Backend** | Not started | Spring Boot. Will handle users, events, payments, marketplace. |
| **Mobile App** | Not started | Kotlin (Android first, iOS planned). Camera tethering + runner search. |
| **Website** | Not started | Next.js on Vercel. Runner search + photographer dashboard. |

**Next priorities:**
1. Finish ai-api production hardening (Phase 7) — event isolation, confidence thresholds
2. Build backend (Spring Boot) — user auth, events, participant management, marketplace
3. Build mobile app and website in parallel

---

## Related Documents

| Document | Location | Content |
|----------|----------|---------|
| ai-api Architecture | `ai-api/docs/architecture.md` | 4-layer design, model registry, async patterns |
| ai-api API Reference | `ai-api/docs/api-reference.md` | All endpoints, request/response examples |
| Integration Architecture | `ai-api/docs/integration-architecture.md` | Responsibility boundary: ai-api vs backends vs desktop |
| Integration Contracts | `ai-api/docs/integration-contracts.md` | Exact API usage patterns per backend, code examples |
| Feature Analysis | `ai-api/docs/feature-analysis-report.md` | Production readiness audit — issues to fix before launch |

---

## Summary

EventAI creates a complete ecosystem for marathon photography in Cebu by combining:

| Problem | Solution |
|---------|----------|
| No real-time upload | Camera → Mobile → Cloud pipeline |
| Manual photo sorting (1-2 hours) | Desktop app auto-sort (5-10 seconds) |
| No blur culling | AI-powered blur detection + classification |
| Runners can't find their photos | Face recognition + bib number search |
| No local marketplace | Website + mobile app with purchase + download |

One platform. Three apps. Photographers save hours. Runners find photos in seconds.
