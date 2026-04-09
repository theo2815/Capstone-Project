# EventAI — Implementation Plan

**Date:** 2026-04-09
**Author:** Development Team
**Source of truth:** `docs/project-vision.md`, `ai-api/docs/integration-contracts.md`, `ai-api/docs/integration-architecture.md`

---

## Table of Contents

1. [What's Already Built](#1-whats-already-built)
2. [Backend Implementation Plan (Spring Boot)](#2-backend-implementation-plan-spring-boot)
3. [Mobile App Implementation Plan (Kotlin)](#3-mobile-app-implementation-plan-kotlin)
4. [Website Implementation Plan (Next.js)](#4-website-implementation-plan-nextjs)
5. [Development Phases](#5-development-phases)
6. [Developer Workflow](#6-developer-workflow)

---

## 1. What's Already Built

Before building anything new, know what exists and works.

| Component | Status | What it provides |
|-----------|--------|------------------|
| `ai-api/` | Complete (Phases 1-6) | Blur detect/classify, face enroll/search/compare, bib OCR, batch processing, webhooks, job polling |
| Desktop App | Built (Electron) | Photographer blur detection + auto-sort (own backend, own DB) |

**ai-api endpoints available for integration** (see `ai-api/docs/api-reference.md` for full details):

| Feature | Endpoints |
|---------|-----------|
| Blur | `POST /blur/detect`, `POST /blur/classify`, batch variants |
| Faces | `POST /faces/enroll`, `POST /faces/search`, `POST /faces/compare`, `POST /faces/detect`, `DELETE /faces/persons/{id}`, batch variants |
| Bibs | `POST /bibs/recognize`, batch variant |
| Jobs | `GET /jobs/{job_id}` |
| Webhooks | `POST /webhooks`, `GET /webhooks`, `DELETE /webhooks/{id}` |
| Health | `GET /health`, `GET /health/ready` |

All endpoints require `X-API-Key` header. Responses use `APIResponse` envelope (`success`, `request_id`, `data`, `error`).

---

## 2. Backend Implementation Plan (Spring Boot)

The backend is the **single public API** for both website and mobile app. It handles users, events, photos, payments, and delegates ML work to ai-api.

### 2.1 Folder Structure

```
backend/
├── src/
│   └── main/
│       ├── java/com/eventai/
│       │   ├── EventAiApplication.java          # Spring Boot entry point
│       │   │
│       │   ├── config/                          # Configuration classes
│       │   │   ├── SecurityConfig.java          # Spring Security + JWT
│       │   │   ├── CorsConfig.java              # CORS for web + mobile
│       │   │   ├── S3Config.java                # AWS S3 client bean
│       │   │   ├── AiApiConfig.java             # ai-api connection settings
│       │   │   ├── WebSocketConfig.java         # Real-time notifications
│       │   │   └── AsyncConfig.java             # Thread pool for async tasks
│       │   │
│       │   ├── controller/                      # REST controllers (thin)
│       │   │   ├── AuthController.java
│       │   │   ├── UserController.java
│       │   │   ├── EventController.java
│       │   │   ├── ParticipantController.java
│       │   │   ├── PhotoController.java
│       │   │   ├── SearchController.java
│       │   │   ├── OrderController.java
│       │   │   ├── WebhookReceiverController.java
│       │   │   ├── AdminController.java
│       │   │   └── HealthController.java
│       │   │
│       │   ├── service/                         # Business logic
│       │   │   ├── AuthService.java
│       │   │   ├── UserService.java
│       │   │   ├── EventService.java
│       │   │   ├── ParticipantService.java
│       │   │   ├── PhotoService.java
│       │   │   ├── PhotoProcessingService.java  # Orchestrates ai-api calls
│       │   │   ├── SearchService.java           # Face + bib search for runners
│       │   │   ├── OrderService.java
│       │   │   ├── PaymentService.java
│       │   │   ├── NotificationService.java
│       │   │   ├── StorageService.java          # S3 upload/download
│       │   │   └── AiApiClient.java             # HTTP client to ai-api
│       │   │
│       │   ├── repository/                      # Spring Data JPA repositories
│       │   │   ├── UserRepository.java
│       │   │   ├── EventRepository.java
│       │   │   ├── ParticipantRepository.java
│       │   │   ├── PhotoRepository.java
│       │   │   ├── PhotoTagRepository.java
│       │   │   ├── OrderRepository.java
│       │   │   └── RefreshTokenRepository.java
│       │   │
│       │   ├── model/                           # JPA entities
│       │   │   ├── User.java
│       │   │   ├── Event.java
│       │   │   ├── Participant.java
│       │   │   ├── Photo.java
│       │   │   ├── PhotoTag.java
│       │   │   ├── Order.java
│       │   │   ├── OrderItem.java
│       │   │   └── RefreshToken.java
│       │   │
│       │   ├── dto/                             # Request/response DTOs
│       │   │   ├── auth/
│       │   │   ├── event/
│       │   │   ├── photo/
│       │   │   ├── search/
│       │   │   ├── order/
│       │   │   └── common/
│       │   │       └── ApiResponse.java         # Standard response envelope
│       │   │
│       │   ├── exception/                       # Custom exceptions + handler
│       │   │   ├── GlobalExceptionHandler.java
│       │   │   ├── ResourceNotFoundException.java
│       │   │   ├── UnauthorizedException.java
│       │   │   └── AiApiUnavailableException.java
│       │   │
│       │   ├── security/                        # JWT + role-based access
│       │   │   ├── JwtTokenProvider.java
│       │   │   ├── JwtAuthenticationFilter.java
│       │   │   └── UserPrincipal.java
│       │   │
│       │   └── util/                            # Utilities
│       │       ├── ImageValidator.java
│       │       └── SlugGenerator.java
│       │
│       └── resources/
│           ├── application.yml                  # Main config
│           ├── application-dev.yml              # Dev overrides
│           ├── application-prod.yml             # Prod overrides
│           └── db/migration/                    # Flyway migrations
│               ├── V1__create_users.sql
│               ├── V2__create_events.sql
│               ├── V3__create_participants.sql
│               ├── V4__create_photos.sql
│               └── V5__create_orders.sql
│
├── src/test/java/com/eventai/                   # Tests mirror src structure
│   ├── controller/
│   ├── service/
│   └── repository/
│
├── build.gradle                                 # Dependencies
├── Dockerfile
├── docker-compose.yml                           # Backend + its PostgreSQL
└── .env.example
```

### 2.2 Modules / Services Breakdown

| Module | Responsibility | Key Dependencies |
|--------|---------------|------------------|
| **Auth** | Register, login (email+password), JWT issue/refresh, password reset | Spring Security, BCrypt, JWT |
| **Users** | Profile CRUD, role management (admin, photographer, runner) | — |
| **Events** | Create/update/delete events, event settings (thresholds, pricing), public gallery listing | — |
| **Participants** | Import participant list (CSV or manual), store bib numbers, link to ai-api person_id after face enrollment | AiApiClient (face enroll) |
| **Photos** | Upload → S3, trigger AI processing pipeline, store metadata + tags, generate watermarked thumbnails | StorageService, PhotoProcessingService |
| **PhotoProcessing** | Orchestrate: blur check → face search → bib OCR → merge tags. Handles single + batch. | AiApiClient |
| **Search** | Runner-facing: selfie face search, bib number lookup. Returns matched photos. | AiApiClient (face search), ParticipantRepository |
| **Orders** | Shopping cart, checkout, payment processing, download links for purchased photos | PaymentService |
| **Payments** | Payment gateway integration (GCash, Maya, card via Stripe/PayMongo) | External payment API |
| **Notifications** | Push notifications (FCM for mobile), WebSocket for real-time web updates | Firebase Cloud Messaging |
| **Storage** | S3 upload (originals, thumbnails, watermarked), presigned URL generation for downloads | AWS S3 SDK |
| **AiApiClient** | HTTP client wrapper for all ai-api calls with retry logic, error mapping, health checks | WebClient/RestClient |

### 2.3 API Endpoints

All endpoints prefixed with `/api/v1/`. Standard response envelope:

```json
{
  "success": true,
  "message": "...",
  "data": { ... },
  "errors": null
}
```

#### Auth

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/auth/register` | Create account (email, password, name, role) | Public |
| POST | `/auth/login` | Login → returns JWT access + refresh token | Public |
| POST | `/auth/refresh` | Refresh access token | Refresh token |
| POST | `/auth/forgot-password` | Send password reset email | Public |
| POST | `/auth/reset-password` | Reset password with token | Public |
| GET | `/auth/me` | Get current user profile | JWT |

#### Users

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/users/me` | Get own profile | JWT |
| PUT | `/users/me` | Update own profile | JWT |
| PUT | `/users/me/avatar` | Upload profile photo | JWT |
| GET | `/users/{id}` | Get public photographer profile | JWT |

#### Events

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/events` | Create event | Admin |
| GET | `/events` | List public events (paginated, filterable) | Public |
| GET | `/events/{id}` | Get event details | Public |
| PUT | `/events/{id}` | Update event settings | Admin |
| DELETE | `/events/{id}` | Delete event (soft) | Admin |
| POST | `/events/{id}/photographers` | Assign photographer to event | Admin |
| GET | `/events/{id}/photographers` | List assigned photographers | Admin |
| GET | `/events/{id}/stats` | Event statistics (photo count, sales, etc.) | Admin/Photographer |

#### Participants

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/events/{id}/participants` | Add single participant | Admin |
| POST | `/events/{id}/participants/import` | Bulk import from CSV (name, bib_number, category) | Admin |
| GET | `/events/{id}/participants` | List participants (paginated) | Admin |
| PUT | `/events/{id}/participants/{pid}` | Update participant | Admin |
| DELETE | `/events/{id}/participants/{pid}` | Remove participant (+ GDPR: delete from ai-api) | Admin |
| POST | `/events/{id}/participants/{pid}/enroll-face` | Upload face photo → enroll in ai-api | Admin |
| POST | `/events/{id}/participants/enroll-face/batch` | Batch face enrollment | Admin |

#### Photos

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/events/{id}/photos` | Upload single photo (triggers AI pipeline) | Photographer |
| POST | `/events/{id}/photos/batch` | Upload multiple photos (async processing) | Photographer |
| GET | `/events/{id}/photos` | Browse event gallery (paginated, watermarked) | Public |
| GET | `/events/{id}/photos/{pid}` | Get single photo details + tags | Public |
| DELETE | `/events/{id}/photos/{pid}` | Delete photo | Photographer/Admin |
| GET | `/events/{id}/photos/processing-status` | Check batch upload processing status | Photographer |

#### Search (Runner-Facing)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/events/{id}/search/face` | Upload selfie → find your photos via face match | JWT (Runner) |
| GET | `/events/{id}/search/bib?number=1023` | Search photos by bib number | JWT (Runner) |
| GET | `/events/{id}/search/results` | Get cached search results for current user | JWT (Runner) |

#### Orders & Payments

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/orders` | Create order (list of photo IDs) | JWT (Runner) |
| GET | `/orders` | List own orders | JWT |
| GET | `/orders/{id}` | Get order details | JWT |
| POST | `/orders/{id}/pay` | Initiate payment | JWT |
| POST | `/webhooks/payment` | Payment gateway callback | Payment provider |
| GET | `/orders/{id}/downloads` | Get download links (presigned S3 URLs) | JWT |

#### Admin

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/admin/dashboard` | Platform stats (events, users, revenue) | Admin |
| GET | `/admin/users` | List all users (paginated) | Admin |
| PUT | `/admin/users/{id}/role` | Change user role | Admin |
| GET | `/admin/events/{id}/sales` | Event sales report | Admin |

#### Internal (ai-api Webhook Receiver)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/internal/ai-webhook` | Receive batch job completion from ai-api | Webhook secret |

### 2.4 Database Schema

Using PostgreSQL (same RDS instance as ai-api, separate database or schema). Flyway for migrations.

```sql
-- V1: Users
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name          VARCHAR(255) NOT NULL,
    role          VARCHAR(20) NOT NULL DEFAULT 'runner',  -- admin, photographer, runner
    avatar_url    VARCHAR(2048),
    is_active     BOOLEAN DEFAULT true,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE refresh_tokens (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- V2: Events
CREATE TABLE events (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                   VARCHAR(255) NOT NULL,
    slug                   VARCHAR(255) UNIQUE NOT NULL,  -- URL-friendly: cebu-marathon-2026
    description            TEXT,
    event_date             DATE NOT NULL,
    location               VARCHAR(255),
    cover_image_url        VARCHAR(2048),
    status                 VARCHAR(20) DEFAULT 'draft',  -- draft, active, completed, archived
    -- AI settings (per-event thresholds, sent to ai-api calls)
    face_match_threshold   FLOAT DEFAULT 0.6,
    bib_confidence_threshold FLOAT DEFAULT 0.7,
    blur_auto_reject       BOOLEAN DEFAULT true,
    -- Pricing
    photo_price            DECIMAL(10,2) DEFAULT 0.00,
    currency               VARCHAR(3) DEFAULT 'PHP',
    -- Metadata
    created_by             UUID REFERENCES users(id),
    created_at             TIMESTAMPTZ DEFAULT now(),
    updated_at             TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE event_photographers (
    event_id        UUID REFERENCES events(id) ON DELETE CASCADE,
    photographer_id UUID REFERENCES users(id) ON DELETE CASCADE,
    assigned_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (event_id, photographer_id)
);

-- V3: Participants
CREATE TABLE participants (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id      UUID REFERENCES events(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    bib_number    VARCHAR(20),
    category      VARCHAR(100),             -- e.g., "21K", "42K", "Fun Run"
    email         VARCHAR(255),             -- optional, for notifications
    ai_person_id  UUID,                     -- ai-api person_id after face enrollment
    enrollment_status VARCHAR(20) DEFAULT 'pending',  -- pending, enrolled, low_quality, failed
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (event_id, bib_number)
);

CREATE INDEX idx_participants_event ON participants(event_id);
CREATE INDEX idx_participants_bib ON participants(event_id, bib_number);
CREATE INDEX idx_participants_ai_person ON participants(ai_person_id);

-- V4: Photos
CREATE TABLE photos (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id           UUID REFERENCES events(id) ON DELETE CASCADE,
    uploaded_by        UUID REFERENCES users(id),
    -- Storage
    original_url       VARCHAR(2048) NOT NULL,    -- S3 key for full-res
    thumbnail_url      VARCHAR(2048),             -- S3 key for thumbnail
    watermarked_url    VARCHAR(2048),             -- S3 key for watermarked preview
    -- Metadata
    filename           VARCHAR(255),
    file_size          BIGINT,
    width              INT,
    height             INT,
    taken_at           TIMESTAMPTZ,               -- EXIF date if available
    -- AI results (persisted from ai-api responses)
    blur_score         FLOAT,                     -- laplacian_variance from blur/detect
    is_blurry          BOOLEAN DEFAULT false,
    blur_type          VARCHAR(50),               -- from blur/classify
    blur_confidence    FLOAT,
    -- Processing status
    processing_status  VARCHAR(20) DEFAULT 'pending',  -- pending, processing, completed, failed
    processing_error   TEXT,
    created_at         TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_photos_event ON photos(event_id);
CREATE INDEX idx_photos_status ON photos(processing_status);

CREATE TABLE photo_tags (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id        UUID REFERENCES photos(id) ON DELETE CASCADE,
    participant_id  UUID REFERENCES participants(id) ON DELETE CASCADE,
    match_method    VARCHAR(10) NOT NULL,    -- 'face' or 'bib'
    confidence      FLOAT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (photo_id, participant_id, match_method)
);

CREATE INDEX idx_photo_tags_photo ON photo_tags(photo_id);
CREATE INDEX idx_photo_tags_participant ON photo_tags(participant_id);

-- V5: Orders
CREATE TABLE orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id),
    event_id        UUID REFERENCES events(id),
    status          VARCHAR(20) DEFAULT 'pending',  -- pending, paid, failed, refunded
    total_amount    DECIMAL(10,2) NOT NULL,
    currency        VARCHAR(3) DEFAULT 'PHP',
    payment_method  VARCHAR(50),
    payment_ref     VARCHAR(255),             -- external payment reference
    paid_at         TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE order_items (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id  UUID REFERENCES orders(id) ON DELETE CASCADE,
    photo_id  UUID REFERENCES photos(id),
    price     DECIMAL(10,2) NOT NULL
);
```

### 2.5 Integration with ai-api

The backend talks to ai-api via HTTP. Key integration class: `AiApiClient.java`.

#### AiApiClient Design

```
AiApiClient
├── checkHealth()                     → GET  /health/ready
├── detectBlur(imageBytes)            → POST /blur/detect
├── classifyBlur(imageBytes)          → POST /blur/classify
├── enrollFace(imageBytes, name, eventId) → POST /faces/enroll
├── searchFace(imageBytes, eventId, threshold, topK) → POST /faces/search
├── searchFaceBatch(files, eventId)   → POST /faces/search/batch
├── recognizeBib(imageBytes)          → POST /bibs/recognize
├── recognizeBibBatch(files)          → POST /bibs/recognize/batch
├── getJobStatus(jobId)               → GET  /jobs/{jobId}
├── deletePerson(personId)            → DELETE /faces/persons/{personId}
└── registerWebhook(url, events)      → POST /webhooks
```

**Retry logic** (from integration-contracts.md):
- 429 → wait `Retry-After` header seconds
- 5xx → exponential backoff (2s, 4s, 8s), max 3 retries
- `MODEL_UNAVAILABLE` → queue for retry, show "AI processing temporarily unavailable"

#### Photo Upload Pipeline

When a photographer uploads a photo, `PhotoProcessingService` runs this pipeline:

```
1. Validate image (format, size)
2. Upload original to S3 → get original_url
3. Generate thumbnail → upload to S3
4. Generate watermarked preview → upload to S3
5. Call ai-api blur/detect
   ├── If blurry + auto_reject → mark photo rejected, stop
   └── If sharp → continue
6. Call ai-api faces/search (with event_id)
   └── For each match → look up participant by ai_person_id → create PhotoTag
7. Call ai-api bibs/recognize
   └── For each detection above bib_confidence_threshold
       → look up participant by bib_number + event_id → create PhotoTag
8. Deduplicate tags (same participant found by face + bib)
9. Mark photo processing_status = 'completed'
10. Send notification to tagged participants (if they have accounts)
```

For batch uploads (200+ photos), use ai-api batch endpoints + webhooks:

```
1. Upload all images to S3
2. Submit blur/detect/batch → get job_id
3. Register webhook for job completion
4. When webhook fires → fetch results
5. Filter out blurry photos
6. Submit faces/search/batch for sharp photos → get job_id
7. Submit bibs/recognize/batch → get job_id
8. When both complete → merge face + bib tags
9. Notify photographer: "200 photos processed, 180 tagged"
```

### 2.6 File Upload Flow (Mobile → Backend → Cloud)

```
Mobile App                          Backend (Spring Boot)               AWS S3
    │                                    │                                │
    │  POST /events/{id}/photos          │                                │
    │  Content-Type: multipart/form-data │                                │
    │  Body: image file (JPEG, ≤10MB)    │                                │
    │───────────────────────────────────►│                                │
    │                                    │                                │
    │                                    │  1. Validate image             │
    │                                    │  2. Generate S3 key:           │
    │                                    │     events/{eventId}/originals/│
    │                                    │     {uuid}.jpg                 │
    │                                    │  3. Upload original            │
    │                                    │───────────────────────────────►│
    │                                    │                                │
    │                                    │  4. Generate thumbnail (800px) │
    │                                    │  5. Upload thumbnail           │
    │                                    │───────────────────────────────►│
    │                                    │                                │
    │                                    │  6. Generate watermarked       │
    │                                    │  7. Upload watermarked         │
    │                                    │───────────────────────────────►│
    │                                    │                                │
    │  202 { photo_id, status: pending } │                                │
    │◄───────────────────────────────────│                                │
    │                                    │                                │
    │                                    │  8. Async: trigger AI pipeline │
    │                                    │     (blur → face → bib)        │
```

**S3 Bucket Structure:**

```
eventai-photos/
├── events/
│   └── {event_id}/
│       ├── originals/       # Full-resolution (private, download after purchase)
│       │   └── {uuid}.jpg
│       ├── thumbnails/      # 800px wide (for gallery browse)
│       │   └── {uuid}.jpg
│       └── watermarked/     # Watermarked preview (public browse)
│           └── {uuid}.jpg
├── avatars/
│   └── {user_id}.jpg
└── participants/
    └── {event_id}/
        └── {participant_id}.jpg   # Enrollment face photo
```

### 2.7 Real-Time Features

#### Push Notifications (Mobile — Firebase Cloud Messaging)

| Trigger | Recipient | Message |
|---------|-----------|---------|
| Photo tagged with participant | Runner (if registered) | "You were spotted in Cebu Marathon! View your photos" |
| Batch processing complete | Photographer | "200 photos processed — 180 tagged" |
| Photo purchased | Photographer | "Someone purchased your photo!" |
| Order ready for download | Runner | "Your photos are ready to download" |

#### WebSocket (Website — Real-Time Updates)

Use Spring WebSocket (STOMP over SockJS) for:

| Channel | Purpose |
|---------|---------|
| `/topic/events/{id}/photos` | Live gallery updates as new photos are processed |
| `/user/queue/notifications` | Personal notifications (photo tagged, order ready) |
| `/topic/events/{id}/upload-progress` | Batch upload progress for photographer |

### 2.8 Security

| Concern | Implementation |
|---------|---------------|
| Authentication | JWT (access token: 15min, refresh token: 7 days). BCrypt password hashing. |
| Authorization | Role-based: Admin, Photographer, Runner. Spring Security `@PreAuthorize`. |
| API key for ai-api | Stored in env vars, never exposed to clients. Backend-to-ai-api only. |
| Input validation | Bean Validation (`@Valid`) on all DTOs. Image validation (type, size, dimensions). |
| CORS | Allow only `https://eventai.ph` (production), `http://localhost:3000` (dev). |
| Rate limiting | Spring Boot rate limiter on public endpoints (login, search). |
| File uploads | Max 10MB per image. Accept only JPEG/PNG/WebP. Validate magic bytes, not just extension. |
| S3 access | Presigned URLs for downloads (expire in 1 hour). Originals never publicly accessible. |
| SQL injection | Spring Data JPA parameterized queries (default). |
| GDPR | `DELETE /participants/{id}` removes from both backend DB and ai-api (face embeddings). |

---

## 3. Mobile App Implementation Plan (Kotlin)

Android-first using Kotlin. Two distinct user flows: Photographer (camera upload) and Runner (search + purchase).

### 3.1 Project Structure (MVVM + Clean Architecture)

```
mobile/
├── app/
│   └── src/main/
│       ├── java/com/eventai/app/
│       │   ├── EventAiApp.kt                    # Application class (DI setup)
│       │   │
│       │   ├── di/                              # Dependency Injection (Hilt)
│       │   │   ├── AppModule.kt
│       │   │   ├── NetworkModule.kt
│       │   │   └── DatabaseModule.kt
│       │   │
│       │   ├── data/                            # Data layer
│       │   │   ├── remote/                      # Network
│       │   │   │   ├── api/
│       │   │   │   │   ├── AuthApi.kt           # Retrofit interface
│       │   │   │   │   ├── EventApi.kt
│       │   │   │   │   ├── PhotoApi.kt
│       │   │   │   │   ├── SearchApi.kt
│       │   │   │   │   └── OrderApi.kt
│       │   │   │   ├── dto/                     # API response models
│       │   │   │   └── interceptor/
│       │   │   │       └── AuthInterceptor.kt   # Attach JWT to requests
│       │   │   │
│       │   │   ├── local/                       # Local storage
│       │   │   │   ├── db/
│       │   │   │   │   ├── AppDatabase.kt       # Room database
│       │   │   │   │   └── dao/
│       │   │   │   │       ├── UploadQueueDao.kt
│       │   │   │   │       └── CachedEventDao.kt
│       │   │   │   └── prefs/
│       │   │   │       └── UserPreferences.kt   # DataStore for tokens, settings
│       │   │   │
│       │   │   └── repository/                  # Repository implementations
│       │   │       ├── AuthRepository.kt
│       │   │       ├── EventRepository.kt
│       │   │       ├── PhotoRepository.kt
│       │   │       ├── SearchRepository.kt
│       │   │       ├── OrderRepository.kt
│       │   │       └── UploadRepository.kt      # Upload queue management
│       │   │
│       │   ├── domain/                          # Domain layer
│       │   │   ├── model/                       # Domain models
│       │   │   │   ├── User.kt
│       │   │   │   ├── Event.kt
│       │   │   │   ├── Photo.kt
│       │   │   │   └── Order.kt
│       │   │   └── usecase/                     # Use cases
│       │   │       ├── LoginUseCase.kt
│       │   │       ├── UploadPhotoUseCase.kt
│       │   │       ├── SearchByFaceUseCase.kt
│       │   │       ├── SearchByBibUseCase.kt
│       │   │       └── PurchasePhotoUseCase.kt
│       │   │
│       │   └── ui/                              # Presentation layer
│       │       ├── theme/                       # Material Design theme
│       │       ├── navigation/
│       │       │   └── AppNavGraph.kt           # Jetpack Navigation
│       │       ├── common/                      # Shared composables
│       │       │   ├── LoadingScreen.kt
│       │       │   ├── ErrorScreen.kt
│       │       │   └── PhotoGrid.kt
│       │       │
│       │       ├── auth/                        # Auth screens
│       │       │   ├── LoginScreen.kt
│       │       │   ├── RegisterScreen.kt
│       │       │   └── AuthViewModel.kt
│       │       │
│       │       ├── photographer/                # Photographer flow
│       │       │   ├── camera/
│       │       │   │   ├── CameraConnectionScreen.kt
│       │       │   │   ├── CameraPreviewScreen.kt
│       │       │   │   └── CameraViewModel.kt
│       │       │   ├── upload/
│       │       │   │   ├── UploadProgressScreen.kt
│       │       │   │   └── UploadViewModel.kt
│       │       │   └── dashboard/
│       │       │       ├── PhotographerDashboardScreen.kt
│       │       │       └── DashboardViewModel.kt
│       │       │
│       │       ├── runner/                      # Runner flow
│       │       │   ├── events/
│       │       │   │   ├── EventListScreen.kt
│       │       │   │   ├── EventDetailScreen.kt
│       │       │   │   └── EventViewModel.kt
│       │       │   ├── search/
│       │       │   │   ├── SearchScreen.kt      # Face selfie + bib input
│       │       │   │   ├── SearchResultsScreen.kt
│       │       │   │   └── SearchViewModel.kt
│       │       │   ├── gallery/
│       │       │   │   ├── GalleryScreen.kt
│       │       │   │   ├── PhotoDetailScreen.kt
│       │       │   │   └── GalleryViewModel.kt
│       │       │   └── purchase/
│       │       │       ├── CartScreen.kt
│       │       │       ├── CheckoutScreen.kt
│       │       │       └── OrderViewModel.kt
│       │       │
│       │       └── profile/
│       │           ├── ProfileScreen.kt
│       │           └── ProfileViewModel.kt
│       │
│       ├── res/                                 # Resources
│       └── AndroidManifest.xml
│
├── build.gradle.kts                             # App-level
└── gradle/                                      # Project-level
```

### 3.2 Core Screens

#### Photographer Flow

| Screen | Purpose | Key Actions |
|--------|---------|-------------|
| Camera Connection | Connect to camera via WiFi or USB tethering | Scan for cameras, pair, show connection status |
| Camera Preview | Live preview of camera feed, auto-upload indicator | See photos as they're taken, upload status per photo |
| Upload Progress | Show upload queue, progress, failures | Retry failed uploads, pause/resume |
| Photographer Dashboard | Event summary, photo count, earnings | Select active event, view stats |

#### Runner Flow

| Screen | Purpose | Key Actions |
|--------|---------|-------------|
| Event List | Browse upcoming/past events | Search, filter by date/location |
| Event Detail | Event info, photo count, search entry point | "Find My Photos" button |
| Search | Face selfie capture OR bib number input | Take selfie with CameraX, enter bib manually |
| Search Results | Grid of matched photos (watermarked) | Select photos, add to cart |
| Photo Detail | Full watermarked preview, tag info | Add to cart, share |
| Cart | Selected photos, total price | Remove items, proceed to checkout |
| Checkout | Payment method selection, confirm | GCash, Maya, card |
| Order History | Past purchases, download links | Re-download purchased photos |

#### Shared Screens

| Screen | Purpose |
|--------|---------|
| Login / Register | Email + password, role selection (photographer/runner) |
| Profile | Edit name, avatar, view account info |
| Notifications | Push notification history |

### 3.3 State Management

| Layer | Tool | Purpose |
|-------|------|---------|
| UI State | Jetpack Compose + `StateFlow` | Screen-level reactive state in ViewModels |
| Navigation | Jetpack Navigation Compose | Type-safe navigation between screens |
| Async | Kotlin Coroutines + Flow | API calls, database queries, upload queue |
| DI | Hilt | Inject repositories, use cases, API clients |
| Local DB | Room | Upload queue (offline), cached events |
| Preferences | DataStore | JWT tokens, user settings, selected event |
| Image Loading | Coil | Async image loading with caching |

### 3.4 API Integration Strategy

```
UI (Compose) → ViewModel → UseCase → Repository → Retrofit API → Backend
                                          ↕
                                    Room (cache/offline)
```

- **Retrofit** with OkHttp for HTTP calls to the Spring Boot backend
- **AuthInterceptor** automatically attaches JWT to every request
- **TokenRefreshAuthenticator** handles 401 → auto-refresh token → retry
- All API calls return `Flow<Resource<T>>` where `Resource` is `Loading | Success | Error`
- Backend base URL configurable via BuildConfig (dev vs. prod)

### 3.5 Camera Connection + Real-Time Upload

This is the most complex mobile feature. Two approaches:

#### Option A: WiFi Tethering (Canon, Sony, Nikon cameras with WiFi)

```
Camera (WiFi AP) ←──WiFi──► Phone ──►4G/WiFi──► Backend ──► S3
```

1. Phone connects to camera's WiFi network
2. Use camera's HTTP API or PTP/IP protocol to receive new photos
3. Each new photo triggers upload to backend via mobile data (or second WiFi if supported)
4. Challenge: phone can't use camera WiFi and internet simultaneously on most devices
5. Solution: Use Android's multi-network API (`ConnectivityManager.requestNetwork`) to bind camera traffic to WiFi and upload traffic to cellular

#### Option B: USB Tethering (via OTG cable)

```
Camera ──USB OTG──► Phone ──►4G/WiFi──► Backend ──► S3
```

1. Camera connected via USB OTG cable
2. Use Android USB Host API + PTP (Picture Transfer Protocol) to receive photos
3. Upload each photo to backend over cellular/WiFi
4. More reliable than WiFi approach (no network conflict)

#### Upload Queue (Both Options)

```kotlin
// Room entity for upload queue
@Entity
data class PendingUpload(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val eventId: String,
    val filePath: String,       // Local file path
    val status: UploadStatus,   // PENDING, UPLOADING, COMPLETED, FAILED
    val retryCount: Int = 0,
    val createdAt: Long = System.currentTimeMillis()
)
```

- **WorkManager** handles upload queue with retry and connectivity constraints
- Photos queued locally even if offline → uploaded when connection restored
- Progress shown in persistent notification
- Failed uploads retry with exponential backoff (max 5 retries)

### 3.6 Search (Face + Bib) UI Flow

#### Face Search Flow

```
1. Runner taps "Find My Photos"
2. Choose "Search by Face"
3. CameraX opens front camera
4. User takes selfie (with face detection overlay to guide framing)
5. App uploads selfie to POST /events/{id}/search/face
6. Backend calls ai-api faces/search with event_id
7. Backend returns matched photos
8. App shows results grid (watermarked thumbnails)
```

#### Bib Search Flow

```
1. Runner taps "Find My Photos"
2. Choose "Search by Bib Number"
3. Numeric keyboard opens
4. User types bib number (e.g., "1023")
5. App calls GET /events/{id}/search/bib?number=1023
6. Backend looks up participant → finds tagged photos
7. App shows results grid
```

### 3.7 Notification System

| Technology | Purpose |
|-----------|---------|
| Firebase Cloud Messaging (FCM) | Push notifications to Android |
| Backend sends via Firebase Admin SDK | Triggered by photo processing, purchase events |
| Local notification channel | Upload progress (foreground service) |

**Backend stores FCM token:**
- On login, mobile app sends FCM token to `PUT /users/me` (or dedicated endpoint)
- Backend stores token in `users.fcm_token` column
- When event triggers notification, backend sends via Firebase Admin SDK

### 3.8 Offline Handling

| Feature | Offline Behavior |
|---------|-----------------|
| Camera upload | Photos queued in Room DB, uploaded when online (WorkManager) |
| Event browsing | Cached event list in Room DB, stale data shown with "offline" banner |
| Search | Not available offline (requires AI processing) |
| Photo gallery | Cached thumbnails via Coil disk cache |
| Cart | Stored locally, synced on next connection |
| Purchases | Not available offline |

---

## 4. Website Implementation Plan (Next.js)

Hosted on Vercel. Server-side rendering for SEO (event pages, public galleries). Client-side for interactive features (search, cart).

### 4.1 Project Structure

```
website/
├── src/
│   ├── app/                                # Next.js App Router
│   │   ├── layout.tsx                      # Root layout (navbar, footer)
│   │   ├── page.tsx                        # Landing page
│   │   ├── globals.css
│   │   │
│   │   ├── (auth)/                         # Auth group (no navbar)
│   │   │   ├── login/page.tsx
│   │   │   ├── register/page.tsx
│   │   │   └── forgot-password/page.tsx
│   │   │
│   │   ├── events/                         # Public event pages
│   │   │   ├── page.tsx                    # Event listing (SSR)
│   │   │   └── [slug]/
│   │   │       ├── page.tsx                # Event detail (SSR)
│   │   │       ├── gallery/page.tsx        # Photo gallery (SSR + client pagination)
│   │   │       └── search/page.tsx         # Search page (client-side)
│   │   │
│   │   ├── dashboard/                      # Protected: Photographer dashboard
│   │   │   ├── layout.tsx                  # Dashboard sidebar layout
│   │   │   ├── page.tsx                    # Overview / stats
│   │   │   ├── events/
│   │   │   │   ├── page.tsx                # My events
│   │   │   │   └── [id]/
│   │   │   │       ├── page.tsx            # Event management
│   │   │   │       ├── upload/page.tsx     # Photo upload
│   │   │   │       └── photos/page.tsx     # Manage photos
│   │   │   └── earnings/page.tsx           # Sales & earnings
│   │   │
│   │   ├── admin/                          # Protected: Admin panel
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx                    # Admin dashboard
│   │   │   ├── events/
│   │   │   │   ├── page.tsx                # Manage all events
│   │   │   │   ├── new/page.tsx            # Create event
│   │   │   │   └── [id]/
│   │   │   │       ├── page.tsx            # Edit event
│   │   │   │       ├── participants/page.tsx  # Manage participants
│   │   │   │       └── settings/page.tsx   # AI thresholds, pricing
│   │   │   └── users/page.tsx              # Manage users
│   │   │
│   │   ├── cart/page.tsx                   # Shopping cart
│   │   ├── checkout/page.tsx               # Payment
│   │   ├── orders/                         # Order history
│   │   │   ├── page.tsx
│   │   │   └── [id]/page.tsx              # Order detail + downloads
│   │   │
│   │   ├── profile/page.tsx                # User profile
│   │   └── api/                            # Next.js API routes (minimal)
│   │       └── auth/[...nextauth]/route.ts # If using NextAuth (optional)
│   │
│   ├── components/                         # Reusable components
│   │   ├── ui/                             # Generic UI primitives
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Badge.tsx
│   │   │   ├── Dropdown.tsx
│   │   │   ├── Pagination.tsx
│   │   │   ├── Spinner.tsx
│   │   │   └── Toast.tsx
│   │   │
│   │   ├── layout/                         # Layout components
│   │   │   ├── Navbar.tsx
│   │   │   ├── Footer.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── DashboardLayout.tsx
│   │   │
│   │   ├── auth/                           # Auth components
│   │   │   ├── LoginForm.tsx
│   │   │   ├── RegisterForm.tsx
│   │   │   └── ProtectedRoute.tsx
│   │   │
│   │   ├── events/                         # Event components
│   │   │   ├── EventCard.tsx
│   │   │   ├── EventGrid.tsx
│   │   │   └── EventHero.tsx
│   │   │
│   │   ├── photos/                         # Photo components
│   │   │   ├── PhotoGrid.tsx
│   │   │   ├── PhotoCard.tsx               # Watermarked thumbnail
│   │   │   ├── PhotoLightbox.tsx           # Full preview modal
│   │   │   └── PhotoUploader.tsx           # Drag-and-drop upload
│   │   │
│   │   ├── search/                         # Search components
│   │   │   ├── FaceSearchUpload.tsx        # Selfie upload + webcam
│   │   │   ├── BibSearchInput.tsx          # Bib number input
│   │   │   └── SearchResults.tsx
│   │   │
│   │   └── cart/                           # Cart components
│   │       ├── CartItem.tsx
│   │       ├── CartSummary.tsx
│   │       └── CheckoutForm.tsx
│   │
│   ├── lib/                                # Utilities and clients
│   │   ├── api.ts                          # API client (fetch wrapper with auth)
│   │   ├── auth.ts                         # JWT token management
│   │   ├── types.ts                        # TypeScript types (mirror backend DTOs)
│   │   └── utils.ts                        # Helpers (formatDate, formatPrice, etc.)
│   │
│   ├── hooks/                              # Custom React hooks
│   │   ├── useAuth.ts                      # Auth state + login/logout
│   │   ├── useCart.ts                      # Cart state management
│   │   ├── useWebSocket.ts                 # Real-time updates
│   │   └── useInfiniteScroll.ts            # Gallery pagination
│   │
│   └── store/                              # Global state (Zustand)
│       ├── authStore.ts
│       └── cartStore.ts
│
├── public/                                 # Static assets
│   └── images/
├── next.config.ts
├── tailwind.config.ts
├── package.json
└── .env.local.example
```

### 4.2 Pages and Feature Breakdown

#### Public Pages (SSR for SEO)

| Page | Route | Rendering | Description |
|------|-------|-----------|-------------|
| Landing | `/` | SSR | Hero, featured events, how it works, CTA |
| Event List | `/events` | SSR | All active events with search/filter |
| Event Detail | `/events/[slug]` | SSR | Event info, photo count, "Find My Photos" CTA |
| Gallery | `/events/[slug]/gallery` | SSR + Client | Paginated watermarked photo grid |
| Search | `/events/[slug]/search` | Client | Face upload + bib input, results display |

#### Protected Pages (Client-Side)

| Page | Route | Role | Description |
|------|-------|------|-------------|
| Login | `/login` | Public | Email + password |
| Register | `/register` | Public | Name, email, password, role selector |
| Cart | `/cart` | Runner | Review selected photos, totals |
| Checkout | `/checkout` | Runner | Payment method, confirm |
| Orders | `/orders` | Runner | Purchase history |
| Order Detail | `/orders/[id]` | Runner | Download links for purchased photos |
| Profile | `/profile` | Any | Edit account info |

#### Photographer Dashboard

| Page | Route | Description |
|------|-------|-------------|
| Dashboard Home | `/dashboard` | Overview: events, photo count, earnings |
| My Events | `/dashboard/events` | List of assigned events |
| Event Manage | `/dashboard/events/[id]` | Event photos, stats |
| Upload | `/dashboard/events/[id]/upload` | Drag-and-drop multi-photo upload |
| Photos | `/dashboard/events/[id]/photos` | Manage uploaded photos |
| Earnings | `/dashboard/earnings` | Revenue breakdown |

#### Admin Panel

| Page | Route | Description |
|------|-------|-------------|
| Admin Home | `/admin` | Platform stats |
| Manage Events | `/admin/events` | CRUD all events |
| Create Event | `/admin/events/new` | Event creation form |
| Edit Event | `/admin/events/[id]` | Settings, thresholds, pricing |
| Participants | `/admin/events/[id]/participants` | Import CSV, enroll faces |
| Event Settings | `/admin/events/[id]/settings` | AI thresholds, pricing |
| Manage Users | `/admin/users` | User list, role management |

### 4.3 API Integration

```typescript
// lib/api.ts — Fetch wrapper
const API_BASE = process.env.NEXT_PUBLIC_API_URL; // e.g., https://api.eventai.ph/api/v1

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getAccessToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options?.headers,
    },
  });

  if (res.status === 401) {
    // Try refresh token
    const refreshed = await refreshAccessToken();
    if (refreshed) return apiFetch<T>(path, options); // Retry
    redirectToLogin();
  }

  const data = await res.json();
  if (!data.success) throw new ApiError(data.errors);
  return data.data;
}
```

**State management:** Zustand for global state (auth, cart). React Query (`@tanstack/react-query`) for server state (events, photos, orders) with caching and background refetching.

### 4.4 Search Functionality

#### Face Search Component

```
1. User clicks "Search by Face"
2. Two options presented:
   a. Upload photo file (drag-and-drop or file picker)
   b. Use webcam (via navigator.mediaDevices.getUserMedia)
3. Preview captured/selected image
4. POST /events/{id}/search/face with image as multipart/form-data
5. Show loading spinner with "Searching..."
6. Display results as photo grid (watermarked)
7. Each result shows: thumbnail, confidence badge, "Add to Cart" button
```

#### Bib Search Component

```
1. User clicks "Search by Bib Number"
2. Input field with numeric validation
3. GET /events/{id}/search/bib?number=1023
4. Display results same as face search
```

### 4.5 Upload Interface for Photographers

```
1. Photographer navigates to /dashboard/events/{id}/upload
2. Drag-and-drop zone accepts JPEG/PNG/WebP (max 10MB each)
3. Client-side validation: file type, file size, image dimensions
4. Upload queue shows each file with progress bar
5. POST /events/{id}/photos/batch (multipart, up to 50 files per request)
6. Backend returns 202 with processing status
7. WebSocket connection shows live processing updates:
   - "Checking blur... 45/50"
   - "Tagging faces... 50/50"
   - "Done: 48 sharp, 2 blurry (rejected)"
8. Results page shows processed photos with detected tags
```

### 4.6 Marketplace (View + Purchase)

```
Browse Gallery → Select Photos → Cart → Checkout → Download

Gallery:
- Infinite scroll watermarked photo grid
- Filter by: participant name, bib number, category
- Each photo shows: watermarked preview, tagged participants, price

Cart:
- Stored in Zustand (persists to localStorage)
- Shows selected photos, total price
- "Remove" and "Clear all" actions

Checkout:
- Payment options: GCash, Maya, Credit Card (via PayMongo or Stripe)
- Order confirmation → redirect to order detail page
- On successful payment webhook → mark order as paid

Download:
- Order detail page shows "Download" button per photo
- Backend generates presigned S3 URL (1-hour expiry) for the original
- One-click download of full-resolution image
```

---

## 5. Development Phases

### Phase 1: Backend Foundation (Weeks 1-2)

**What to build:**
- Spring Boot project scaffolding with Gradle
- PostgreSQL connection with Flyway migrations (V1-V5)
- JPA entities: User, Event, Participant, Photo, PhotoTag, Order, OrderItem
- Global exception handler and API response envelope
- Health check endpoint
- Docker Compose for backend (PostgreSQL, Redis, Spring Boot)
- `.env` configuration pattern

**Expected output:**
- `POST /health` returns 200
- Database tables created via Flyway
- Docker Compose starts the backend with all infrastructure

**Dependencies:** None

---

### Phase 2: Authentication System (Weeks 2-3)

**What to build:**
- JWT token provider (access + refresh tokens)
- Spring Security configuration (stateless JWT filter)
- Auth endpoints: register, login, refresh, forgot-password, reset-password
- Role-based access control (Admin, Photographer, Runner)
- Password hashing with BCrypt
- Auth middleware: `@PreAuthorize("hasRole('ADMIN')")` on protected endpoints

**Expected output:**
- Register → Login → Get JWT → Access protected endpoints
- Refresh token flow works
- Role-based access enforced

**Dependencies:** Phase 1

---

### Phase 3: Event + Participant Management (Weeks 3-4)

**What to build:**
- Event CRUD endpoints (create, list, get, update, delete)
- Event slug generation for clean URLs
- Participant CRUD (single add, CSV bulk import)
- Event-photographer assignment
- Event settings (AI thresholds, pricing)

**Expected output:**
- Admin can create event "Cebu Marathon 2026"
- Admin can import 500 participants from CSV
- Admin can assign photographers to events

**Dependencies:** Phase 2

---

### Phase 4: Photo Upload + Storage (Weeks 4-6)

**What to build:**
- AWS S3 integration (StorageService)
- Photo upload endpoint (single + batch)
- Image validation (type, size, magic bytes)
- Thumbnail generation (800px width)
- Watermark generation (overlay text/image)
- Photo metadata extraction (EXIF date, dimensions)
- Photo gallery endpoint (paginated, filterable)

**Expected output:**
- Photographer uploads photo → stored in S3 (original + thumbnail + watermarked)
- Gallery endpoint returns paginated watermarked photos
- Batch upload accepts up to 50 photos

**Dependencies:** Phase 3

---

### Phase 5: AI Integration (Weeks 6-8)

**What to build:**
- `AiApiClient.java` — HTTP client with retry logic
- Health check integration (periodic ai-api health polling)
- Photo processing pipeline: blur → face → bib → merge tags
- Webhook receiver for batch job completion
- Participant face enrollment (`POST /faces/enroll` with `event_id`)
- PhotoTag creation from face matches + bib matches
- Batch processing flow for bulk uploads
- Error handling per integration contracts (429, 503, 5xx)

**Expected output:**
- Upload photo → auto blur check → auto face tag → auto bib tag
- Admin enrolls participant face → stored in ai-api with event_id
- Batch upload 200 photos → async processing → webhook → tags stored
- Failed ai-api calls retry with backoff

**Dependencies:** Phase 4, ai-api running

---

### Phase 6: Mobile App Core (Weeks 6-10)

*Can run in parallel with Phase 5 once backend auth + events + photos are ready.*

**What to build (split into sub-phases):**

**6a — Foundation (Week 6-7):**
- Project setup: Kotlin, Jetpack Compose, Hilt, Room, Retrofit
- Navigation graph
- Auth screens (login, register)
- API client with JWT interceptor + token refresh

**6b — Runner Flow (Week 7-9):**
- Event list + detail screens
- Gallery browse (watermarked photos, infinite scroll)
- Face search screen (CameraX selfie capture + upload)
- Bib search screen (numeric input)
- Search results display
- Cart + checkout screens
- Order history + photo download
- Push notifications (FCM setup)

**6c — Photographer Flow (Week 9-10):**
- Camera connection (USB OTG via PTP protocol — start with this)
- Upload queue with WorkManager (offline-capable)
- Upload progress screen
- Photographer dashboard (event stats)

**Expected output:**
- Runner can browse events, search by face/bib, purchase photos
- Photographer can connect camera and auto-upload photos
- Push notifications work for photo tagging and order completion

**Dependencies:** Phase 2 (auth), Phase 4 (photos), Phase 5 (search)

---

### Phase 7: Website Platform (Weeks 7-10)

*Can run in parallel with Phase 6.*

**What to build (split into sub-phases):**

**7a — Foundation + Public Pages (Week 7-8):**
- Next.js project setup with App Router
- Tailwind CSS + component library (shadcn/ui recommended)
- Landing page
- Event listing page (SSR)
- Event detail page (SSR)
- Photo gallery (SSR + client-side infinite scroll)
- Auth pages (login, register)
- API client with JWT management

**7b — Search + Marketplace (Week 8-9):**
- Face search component (file upload + webcam capture)
- Bib search component
- Search results display
- Cart (Zustand + localStorage persistence)
- Checkout page (payment integration)
- Order history + download pages

**7c — Photographer Dashboard + Admin (Week 9-10):**
- Photographer dashboard (events, upload, manage photos)
- Drag-and-drop photo uploader with progress
- Admin panel (event management, participant import, user management)
- WebSocket integration for real-time updates

**Expected output:**
- Public visitors can browse events and galleries
- Runners can search by face/bib, purchase, and download
- Photographers can upload and manage photos
- Admins can manage events, participants, and users

**Dependencies:** Phase 2 (auth), Phase 4 (photos), Phase 5 (search)

---

### Phase 8: Marketplace + Payments (Weeks 10-12)

**What to build:**
- Payment gateway integration (PayMongo recommended for Philippines — supports GCash, Maya, card)
- Order creation and payment flow
- Payment webhook handling (confirm payment → mark order paid)
- Presigned S3 URL generation for purchased photo downloads
- Download tracking
- Photographer earnings calculation
- Basic sales reporting for admin

**Expected output:**
- Runner completes purchase via GCash/Maya/card
- Payment confirmed via webhook
- Runner can download full-resolution photos
- Photographer sees earnings

**Dependencies:** Phase 7b (cart/checkout UI), Phase 4 (S3)

---

### Phase 9: Notifications + Real-Time (Weeks 11-12)

**What to build:**
- Firebase Cloud Messaging integration in backend
- FCM token registration endpoint
- Notification triggers (photo tagged, order ready, batch complete)
- WebSocket (STOMP) for website real-time updates
- Email notifications (optional — Mailgun/SendGrid for order receipts)

**Expected output:**
- Runner gets push notification when tagged in a photo
- Photographer gets upload progress in real-time on web
- Order confirmation emails sent

**Dependencies:** Phase 6 (mobile FCM), Phase 7 (web WebSocket)

---

### Phase 10: Testing + Polish + Deploy (Weeks 12-14)

**What to build:**
- End-to-end testing (backend + mobile + web)
- Performance testing for photo upload pipeline
- Security audit (OWASP top 10 check)
- Error monitoring setup (Sentry)
- Production deployment:
  - Backend: AWS EC2 (alongside ai-api)
  - Website: Vercel
  - Mobile: Google Play Store (internal testing track)
- NGINX reverse proxy configuration
- SSL certificates (Let's Encrypt)
- Production environment variables
- Backup strategy (RDS automated backups)

**Expected output:**
- All components deployed and accessible
- End-to-end flow works: photographer uploads → AI processes → runner finds → purchases → downloads
- Monitoring and alerting in place

**Dependencies:** All previous phases

---

### Phase Summary Timeline

```
Week  1  2  3  4  5  6  7  8  9  10  11  12  13  14
      ├──┤                                              Phase 1: Backend Foundation
         ├──┤                                           Phase 2: Auth System
            ├──┤                                        Phase 3: Events + Participants
               ├─────┤                                  Phase 4: Photo Upload + Storage
                     ├─────┤                            Phase 5: AI Integration
                     ├──────────────┤                   Phase 6: Mobile App
                        ├──────────────┤                Phase 7: Website
                                    ├─────┤             Phase 8: Payments
                                       ├──┤            Phase 9: Notifications
                                             ├─────┤   Phase 10: Testing + Deploy
```

**Critical path:** Phase 1 → 2 → 3 → 4 → 5 → 8 → 10

**Parallel tracks after Phase 4:**
- Track A: Backend AI integration (Phase 5)
- Track B: Mobile app (Phase 6)
- Track C: Website (Phase 7)

---

## 6. Developer Workflow

### 6.1 How to Develop Backend, Mobile, and Web in Parallel

Once Phase 4 (Photo Upload) is solid, three developers/teams can work in parallel:

```
Developer/Team A: Backend                    Starts: Week 1
├── Phase 1-5: Core backend                  Weeks 1-8
├── Phase 8: Payments                        Weeks 10-12
└── Phase 9: Notifications                   Weeks 11-12

Developer/Team B: Mobile (Kotlin)            Starts: Week 6
├── Phase 6a: Foundation + Auth              Weeks 6-7 (uses backend auth API)
├── Phase 6b: Runner flow                    Weeks 7-9
└── Phase 6c: Photographer flow              Weeks 9-10

Developer/Team C: Website (Next.js)          Starts: Week 7
├── Phase 7a: Public pages + Auth            Weeks 7-8
├── Phase 7b: Search + Marketplace           Weeks 8-9
└── Phase 7c: Dashboard + Admin              Weeks 9-10
```

**Key coordination points:**
1. Backend publishes API docs (Swagger/OpenAPI) as soon as endpoints are built — mobile/web code against them
2. Backend deploys to a shared dev server (or localhost with Docker Compose) for integration testing
3. Mobile and web use mock data while waiting for backend endpoints

### 6.2 Development Environment Setup

```bash
# Terminal 1 — ai-api (already built)
cd ai-api
docker compose up db redis -d
make dev          # FastAPI on :8000
make celery       # Celery worker

# Terminal 2 — Backend
cd backend
./gradlew bootRun  # Spring Boot on :8080

# Terminal 3 — Website
cd website
npm run dev        # Next.js on :3000

# Terminal 4 — Mobile
# Open in Android Studio, run on emulator or device
# Points to http://10.0.2.2:8080 (emulator → host)
```

**Environment files:**

```bash
# backend/.env
DATABASE_URL=jdbc:postgresql://localhost:5432/eventai_backend
REDIS_URL=redis://localhost:6379/1
AI_API_URL=http://localhost:8000
AI_API_KEY=sk_webmobile_dev_xxxxx
JWT_SECRET=dev-secret-change-in-prod
S3_BUCKET=eventai-photos-dev
S3_REGION=ap-southeast-1
PAYMENT_SECRET_KEY=pk_test_xxxxx

# website/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8080/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws

# mobile (BuildConfig)
API_BASE_URL=http://10.0.2.2:8080/api/v1   # Android emulator → host
```

### 6.3 Suggested Milestones

| Milestone | Definition of Done | Target |
|-----------|-------------------|--------|
| **M1: Backend MVP** | Auth + Events + Photos upload + AI tagging works end-to-end via Postman | Week 8 |
| **M2: Runner Search** | Runner can search by face/bib on web and get results | Week 9 |
| **M3: Mobile Upload** | Photographer can upload photos from phone to backend | Week 9 |
| **M4: Marketplace** | Runner can browse, search, purchase, and download photos | Week 12 |
| **M5: Production Deploy** | All components deployed, end-to-end flow works | Week 14 |

### 6.4 Testing Strategy

Keep it simple. Focus on the flows that matter most.

#### Backend Tests

| Type | Scope | Tool | Priority |
|------|-------|------|----------|
| Unit | Services (PhotoProcessingService, AiApiClient) | JUnit 5 + Mockito | High |
| Integration | Controllers + DB (full Spring context) | @SpringBootTest + Testcontainers (PostgreSQL) | High |
| API contract | ai-api client (mock ai-api responses) | WireMock | Medium |

```bash
# Run from backend/
./gradlew test                    # All tests
./gradlew test --tests "*PhotoService*"  # Single test class
```

#### Mobile Tests

| Type | Scope | Tool | Priority |
|------|-------|------|----------|
| Unit | ViewModels, UseCases, Repositories | JUnit + Mockk | High |
| UI | Critical screens (search, cart) | Compose Testing | Medium |

#### Website Tests

| Type | Scope | Tool | Priority |
|------|-------|------|----------|
| Unit | Utility functions, hooks | Vitest | Medium |
| Component | Key components (PhotoUploader, SearchResults) | React Testing Library | Medium |
| E2E | Critical flow: search → cart → checkout | Playwright | Low (add later) |

#### Integration Test Checklist

Before each milestone, verify these end-to-end flows manually:

```
□ Register → Login → Get JWT → Access protected endpoint
□ Create event → Import participants → Enroll faces via ai-api
□ Upload photo → Blur check → Face tag → Bib tag → Photo visible in gallery
□ Runner: Selfie search → Results → Add to cart → Checkout → Download
□ Batch upload 50 photos → All processed → All tagged
□ Photographer: Camera → Upload → See photos in event gallery
```

### 6.5 Branch Strategy

```
main                    ← production-ready, deployed
├── develop             ← integration branch
├── feature/backend-auth
├── feature/backend-photos
├── feature/mobile-runner-flow
├── feature/web-search
└── hotfix/fix-upload-bug
```

- Feature branches off `develop`
- PR review before merge to `develop`
- `develop` → `main` when milestone is complete
- Hotfix branches off `main` for critical production fixes

---

## Appendix A: Key Dependencies

### Backend (Spring Boot)

```groovy
// build.gradle
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
    implementation 'org.springframework.boot:spring-boot-starter-security'
    implementation 'org.springframework.boot:spring-boot-starter-validation'
    implementation 'org.springframework.boot:spring-boot-starter-websocket'
    implementation 'org.flywaydb:flyway-core'
    implementation 'org.postgresql:postgresql'
    implementation 'io.jsonwebtoken:jjwt-api:0.12.5'
    implementation 'software.amazon.awssdk:s3:2.25.0'
    implementation 'com.google.firebase:firebase-admin:9.2.0'
    // Image processing
    implementation 'net.coobird:thumbnailator:0.4.20'
    // Payment (PayMongo)
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'

    runtimeOnly 'io.jsonwebtoken:jjwt-impl:0.12.5'
    runtimeOnly 'io.jsonwebtoken:jjwt-jackson:0.12.5'

    testImplementation 'org.springframework.boot:spring-boot-starter-test'
    testImplementation 'org.testcontainers:postgresql:1.19.7'
    testImplementation 'com.github.tomakehurst:wiremock-jre8:3.0.1'
}
```

### Mobile (Kotlin/Android)

```kotlin
// Key dependencies
// Jetpack Compose + Material 3
// Hilt (DI)
// Retrofit + OkHttp (networking)
// Room (local DB)
// DataStore (preferences)
// Coil (image loading)
// CameraX (selfie capture)
// WorkManager (background uploads)
// Firebase Cloud Messaging
```

### Website (Next.js)

```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.0.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.0.0",
    "tailwindcss": "^3.0.0",
    "next-auth": "^5.0.0"
  }
}
```

---

## Appendix B: Quick Reference — What Calls What

```
┌──────────────────────────────────────────────────────────────┐
│                     CLIENT APPS                              │
│                                                              │
│  Mobile App (Kotlin)  ──┐                                    │
│  Website (Next.js)    ──┼──► Backend (Spring Boot) :8080     │
│                         │    ├── Users, Auth (JWT)            │
│                         │    ├── Events, Participants         │
│                         │    ├── Photos → S3                  │
│                         │    ├── Orders, Payments             │
│                         │    └── Delegates ML to ai-api ─────┤
│                         │                                    │
│                         │    ai-api (FastAPI) :8000           │
│                         │    ├── Blur detect/classify         │
│                         │    ├── Face enroll/search/compare   │
│                         │    ├── Bib OCR                      │
│                         │    └── Batch processing (Celery)    │
│                                                              │
│  Desktop App (Electron) ──► Desktop Backend ──► ai-api       │
│  (already built)            (blur only)                      │
└──────────────────────────────────────────────────────────────┘
```

**Rule: Mobile and Website NEVER talk to ai-api directly. Always through the Spring Boot backend.**
