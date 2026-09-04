---
paths:
  - "backend/**"
  - "website/**"
  - "mobile/**"
---

# ai-api Integration Boundary (loads when touching backend / website / mobile)

Detailed enforcement of the module boundaries. The one-line invariants live in the root `CLAUDE.md`; this file carries the contract you need while actually editing these surfaces.

- **Clients never call `ai-api` directly.** Mobile and website call the Spring Boot backend, which holds the `ai-api` key and proxies inference. (Desktop — an external repo — is the sole exception.)
- **The backend has TWO face/bib providers.** `AiApiClient` and `RekognitionAiClient` both implement `FaceBibProvider`, selected by `AI_PROVIDER` (default `ai-api`; alt `rekognition` → AWS Rekognition). Docs that describe only "backend → ai-api" are incomplete — the AWS path is real.
- **Blur is desktop-only.** No blur endpoint call, no `BLUR_REJECTED` gate, no `blurScore` write in any web/mobile/backend upload path. `AiApiClient` has no `blurDetect()` — do not add one. (Note: `blur_score` survives as an unused column in `backend/.../db/migration/V3__photos.sql`; nothing writes it — leave it unless asked to clean it up.)
- **`AI_API_ENABLED=false` is the dev default** (`backend/.../application.yml`). Master switch for every server-side ai-api call; off → `aiDetectionStatus="none"`, face-search returns 503.
- **Event isolation:** every `faces/enroll` / `faces/search` call passes a non-null `event_id`.
- **Confidence thresholds are the backend's**, defined once in `AiApiProperties.kt` / `application.yml` (face + bib defaults). Read the values there — do not copy them into other docs.

Full contract: `ai-api/docs/integration-contracts.md`. Rationale for these decisions: vault `backend/decisions.md`.
