---
paths:
  - "backend/**"
  - "website/**"
  - "mobile/**"
---

# ai-api Integration Boundary (loads when touching backend / website / mobile)

Detailed enforcement of the module boundaries. The one-line invariants live in the root `CLAUDE.md`; this file carries the contract you need while actually editing these surfaces.

- **Clients never call `ai-api` directly.** Mobile and website call the Spring Boot backend, which does face/bib through AWS Rekognition. (Desktop — an external repo — is the sole caller of `ai-api`, for blur.)
- **One exception: the photographer's live selfie.** The backend creates the AWS Face Liveness session and alone reads its result and decides. Only the live-selfie video goes browser → AWS (`StartFaceLivenessSession`, ap-southeast-5), with short-lived STS keys the backend issues with each session (`LIVENESS_CREDENTIAL_SECONDS`, default 900), limited to that one action in that one region (IAM cannot scope them to one session id). No score ever comes from the browser. Website component: `website/src/components/dashboard/live-selfie-check.tsx`; backend: `LivenessSessionService.kt`.
- **The backend has TWO face/bib providers.** `RekognitionAiClient` and `AiApiClient` both implement `FaceBibProvider`, selected by `AI_PROVIDER` (default `rekognition`, what staging and production run). ai-api's face/bib routes were removed on 2026-10-16, so `AI_PROVIDER=ai-api` gets 404s; `AiApiClient` stays until someone removes that provider.
- **Blur is desktop-only.** No blur endpoint call, no `BLUR_REJECTED` gate, no `blurScore` write in any web/mobile/backend upload path. `AiApiClient` has no `blurDetect()` — do not add one. One exception, after upload only: the Photo Relevance batch check may read Rekognition DetectLabels sharpness for a batch the photographer has stopped uploading to, and stores it in `relevance_photo_checks`, never in `blurScore` / `blur_score`. (Note: `blur_score` survives as an unused column in `backend/.../db/migration/V3__photos.sql`; nothing writes it — leave it unless asked to clean it up.)
- **`AI_API_ENABLED=false` is the dev default** (`backend/.../application.yml`). Master switch for every server-side face/bib call, whichever provider; off → `aiDetectionStatus="none"`, face-search returns 503.
- **Event isolation:** every face enroll / search is scoped to one event (Rekognition: one collection per event).
- **Confidence thresholds are the backend's**, defined once in `AiApiProperties.kt` / `RekognitionProperties.kt` / `PhotoRemovalProperties.kt` / `IdentityProperties.kt` / `ModerationProperties.kt` / `RelevanceProperties.kt` / `application.yml` (face + bib defaults, runner deletion bands, liveness + face-match + image-moderation bars, relevance copy and date rules). Read the values there — do not copy them into other docs.

Full contract: `ai-api/docs/integration-contracts.md`. Rationale for these decisions: vault `backend/decisions.md`.
