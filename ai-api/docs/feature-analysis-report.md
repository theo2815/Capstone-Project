# Feature Analysis Report — EventAI API

**Date:** 2026-03-26
**Scope:** Blur Detection, Face Search, Bib Number Recognition
**Purpose:** Identify issues for production readiness. No fixes proposed.
**Prior audits:** `deep-audit-report.md`, `rescan-audit-report.md` (all findings implemented)

---

## Conventions

| Tag | Meaning |
|-----|---------|
| **Severity** | CRITICAL / HIGH / MEDIUM / LOW / INFO |
| **Owner** | `ai-api` = this codebase, `backend` = upstream backend service, `both` = shared responsibility |

---

## 1. Face Search

### FA-1 — No event or gallery isolation (CRITICAL) `ai-api + backend`

**What:** Face search is isolated by `api_key_id` only. The `Person` model has no `event_id` or `gallery_id` column. Every search query filters solely by `WHERE p.api_key_id = :api_key_id`.

**Observed behavior:** If a single API key is used across multiple events (Marathon A and Triathlon B), enrolling "John" at Marathon A causes John to appear as a match when searching photos from Triathlon B. All persons under one API key share a single embedding space.

**Affected files:**
- `src/db/models.py:16-34` — `Person` model has only `api_key_id`, no event/gallery columns
- `src/db/repositories/face_repo.py:62-108` — `search_similar()` filters by `api_key_id` only
- `src/db/repositories/face_repo.py:110-172` — `batch_search_similar()` same filter
- `src/db/repositories/sync_face_repo.py:13-52` — sync variant, same filter
- `src/api/v1/faces.py:84-195` — enroll endpoint creates Person with `api_key_id` only

**Impact:** Cross-event contamination in search results for any client that reuses an API key across events. This is the most significant production-readiness gap in the face search feature.

**Architecture note:** Whether `event_id` lives in ai-api's DB or is passed as a parameter from the backend is a design decision that spans both codebases. The backend must know about events; ai-api must filter by them.

---

### FA-2 — No bulk enrollment endpoint (MEDIUM) `ai-api`

**What:** The `/faces/enroll` endpoint accepts a single image. There is no batch enrollment endpoint. For a typical event with 500+ participants, enrolling one image at a time is slow and rate-limit-unfriendly.

**Affected files:**
- `src/api/v1/faces.py:84-195` — only single-image `enroll_face()` exists

**Impact:** Clients must loop single-image enrollment calls, incurring per-request overhead (auth, DB session, model load check) for each photo. This is a usability gap for production event setup.

---

### FA-3 — Batch face search does not use batch DB query (LOW) `ai-api`

**What:** The single-image `/faces/search` endpoint uses `batch_search_similar()` to query all faces in one DB round-trip. However, the Celery batch worker (`face_process_batch` → `_search_single`) calls `repo.search_similar()` individually per face per image in a loop.

**Affected files:**
- `src/workers/tasks/face_tasks.py:72-102` — `_search_single()` does N individual queries
- `src/db/repositories/sync_face_repo.py` — no `batch_search_similar()` method exists

**Impact:** A batch of 20 images with 3 faces each generates 60 individual DB queries instead of a smaller number of batched queries. Performance gap, not a correctness issue.

---

### FA-4 — Compare endpoint only uses first face from each image (INFO) `ai-api`

**What:** `/faces/compare` extracts embeddings from both images but only compares `faces1[0]` with `faces2[0]`. If either image has multiple faces, all faces beyond the first are silently ignored.

**Affected files:**
- `src/api/v1/faces.py:313-319` — `emb1 = faces1[0]`, `emb2 = faces2[0]`

**Impact:** If a group photo is submitted for comparison, the result depends on which face InsightFace returns first (typically the largest/most centered). No error or warning is returned to the caller.

---

### FA-5 — Batch face search hardcodes top_k=10 and uses global threshold (LOW) `ai-api`

**What:** The batch face search Celery task uses `settings.FACE_SIMILARITY_THRESHOLD` and hardcodes `top_k=10`. The single-image endpoint allows per-request `threshold` (0.0-1.0) and `top_k` (1-100) query parameters. Batch callers cannot customize these.

**Affected files:**
- `src/workers/tasks/face_tasks.py:88-91` — hardcoded `top_k=10`, uses `settings.FACE_SIMILARITY_THRESHOLD`
- `src/api/v1/faces.py:406-441` — batch endpoint signature has no `threshold` or `top_k` params

**Impact:** Batch and single-image endpoints produce different results for the same image when the caller uses non-default threshold/top_k values.

---

### FA-6 — No duplicate-person or duplicate-embedding guard on enrollment (MEDIUM) `ai-api`

**What:** Enrolling the same face image for the same person multiple times creates duplicate `FaceEmbedding` rows. The `source_image_hash` column is indexed but never checked for uniqueness during enrollment.

**Affected files:**
- `src/api/v1/faces.py:162-168` — stores embedding without checking if `source_image_hash` already exists for `person_id`
- `src/db/repositories/face_repo.py:45-60` — `store_embedding()` has no duplicate check

**Impact:** Duplicate embeddings inflate search results. The same person can appear multiple times in a single search response with near-identical similarity scores.

---

### FA-7 — Persons list endpoint missing (MEDIUM) `ai-api`

**What:** There is a GET `/faces/persons/{person_id}` and DELETE `/faces/persons/{person_id}`, but no GET `/faces/persons` (list all enrolled persons). Clients cannot discover who is enrolled without already knowing person IDs.

**Affected files:**
- `src/api/v1/faces.py` — no list-persons endpoint exists

**Impact:** Operational gap for production. Event organizers cannot audit their enrolled gallery without external tracking.

---

## 2. Bib Number Recognition

### BIB-1 — No bib number search/lookup endpoint (HIGH) `ai-api`

**What:** The API only provides bib number *recognition* (OCR from image). There is no endpoint to search recognized bib numbers or look up results by bib number text. The API returns OCR results, but downstream matching against an event participant list is entirely the caller's responsibility.

**Affected files:**
- `src/api/v1/bibs.py` — only `recognize_bibs()` and `recognize_bibs_batch()` exist

**Impact:** This is not necessarily a bug — it may be by design if the backend handles participant lookup. However, there is no documentation clarifying this boundary.

**Architecture note:** If the backend matches recognized bib numbers to participant records, this is expected. If ai-api is expected to provide end-to-end bib search (image → participant), this is a missing feature.

---

### BIB-2 — No confidence threshold on OCR output (HIGH) `ai-api`

**What:** The `BibRecognizer.recognize()` method returns the best candidate regardless of confidence. A recognition with 0.05 confidence is returned identically to one with 0.95. There is no configurable floor.

**Affected files:**
- `src/ml/bibs/recognizer.py:60-74` — candidates sorted by confidence, best returned with no minimum check
- `src/api/v1/bibs.py:76` — `if ocr_result["bib_number"]` checks for non-empty string only, not confidence
- `src/workers/tasks/bib_tasks.py:48-49` — same: no confidence check
- `src/config.py` — no `BIB_MIN_CONFIDENCE` setting exists

**Impact:** Low-confidence OCR results (wrong digits, partial reads) are presented as valid bib detections. Callers must implement their own confidence filtering, but nothing in the API documentation or response schema signals that this is expected.

---

### BIB-3 — Single-digit bib numbers silently rejected (MEDIUM) `ai-api`

**What:** `BibRecognizer` filters candidates where `digit_count < min_chars` (default 2). Bib numbers "1" through "9" are rejected as invalid even when correctly recognized.

**Affected files:**
- `src/ml/bibs/recognizer.py:63-64` — `if digit_count >= self.min_chars`
- `src/config.py:44` — `BIB_MIN_CHARS: int = 2`

**Impact:** Events using single-digit bib numbers (e.g., elite runners 1-9) get empty results with no error indicator. The response shows `bib_number: ""` with no explanation of why a valid recognition was discarded.

---

### BIB-4 — YOLO detection confidence hardcoded at 0.5 (LOW) `ai-api`

**What:** `BibDetector.detect()` uses `confidence=0.5` as a default method parameter. This is not configurable via settings or API query parameters.

**Affected files:**
- `src/ml/bibs/detector.py:43` — `def detect(self, image, confidence: float = 0.5)`
- `src/api/v1/bibs.py:60` — calls `bib_detector.detect(image)` without passing confidence
- `src/workers/tasks/bib_tasks.py:64` — same: no confidence override

**Impact:** Cannot tune detection sensitivity without code changes. Small or partially occluded bibs below 0.5 confidence are missed with no recourse.

---

### BIB-5 — OCR character substitution not handled (MEDIUM) `ai-api`

**What:** The regex `[A-Za-z0-9\-_]` preserves letters in recognized bib text. Common OCR confusions like "O" → "0", "I" → "1", "l" → "1" are not corrected. The bib number "1O23" (letter O) and "1023" (digit 0) are treated as different results.

**Affected files:**
- `src/ml/bibs/recognizer.py:12` — `_BIB_CHAR_RE = re.compile(r"[A-Za-z0-9\-_]")`
- `src/ml/bibs/recognizer.py:62` — cleaned text preserves mixed alpha/numeric

**Impact:** OCR returns bib numbers with letter/digit confusion that downstream exact-match lookups will fail on.

---

### BIB-6 — Fallback mode (no YOLO) runs OCR on full image (MEDIUM) `ai-api`

**What:** When `BibDetector` is unavailable or its model is not loaded, both the API endpoint and the batch worker run OCR on the entire image rather than failing gracefully. The full-image OCR will pick up any text in the scene (sponsor logos, timing boards, spectator signs) and return it as a bib number candidate.

**Affected files:**
- `src/api/v1/bibs.py:87-100` — fallback OCR on full image with bbox set to entire frame
- `src/workers/tasks/bib_tasks.py:83-92` — `_recognize_single` same fallback

**Impact:** Without the detector, false positives from non-bib text in the scene. The full-frame bbox `{x1: 0, y1: 0, x2: width, y2: height}` gives the caller no spatial information to verify the detection.

---

### BIB-7 — Batch bib recognition ignores api_key_id (INFO) `ai-api`

**What:** The `bib_recognize_batch.delay()` call does not pass `api_key_id` to the Celery task, unlike the face batch endpoint which was fixed in RS-1. Bib recognition is stateless (no DB lookup), so this is not currently a bug. However, if bib recognition ever stores results per-tenant, the plumbing is missing.

**Affected files:**
- `src/api/v1/bibs.py:145` — `bib_recognize_batch.delay(job_id, image_data_list)` — no `api_key_id`
- `src/workers/tasks/bib_tasks.py:10` — task signature has no `api_key_id` param

**Impact:** No current impact since bib recognition is stateless. Future concern only.

---

## 3. Blur Detection

### BLUR-1 — Laplacian variance is image-size-dependent (MEDIUM) `ai-api`

**What:** Laplacian variance increases with image resolution. A 4096x4096 sharp image and a 640x640 sharp image produce different variance values, but the same threshold (default 100.0) is applied. Images are downscaled to `MAX_INFERENCE_DIMENSION` (2048) before inference, but this still means different-sized inputs below 2048px produce inconsistent scores.

**Affected files:**
- `src/ml/blur/detector.py:44-66` — variance computed on whatever size arrives
- `src/utils/image_utils.py:89-106` — downscale only if exceeds MAX_INFERENCE_DIMENSION

**Impact:** Small images (e.g., thumbnails, cropped regions) may be misclassified as blurry because their Laplacian variance is naturally lower. The threshold does not normalize for resolution.

---

### BLUR-2 — FFT high-frequency ring radius may be zero for small images (LOW) `ai-api`

**What:** The FFT mask radius is `r = min(h, w) // 8`. For a 32x32 image (the minimum allowed by `MIN_DIMENSION`), `r = 4`. For an image slightly larger than 32px, the center mask may be so small that `hf_ratio` approaches 1.0 regardless of actual frequency content.

**Affected files:**
- `src/ml/blur/detector.py:60-62` — `r = min(h, w) // 8`

**Impact:** `hf_ratio` metric is unreliable for images near `MIN_DIMENSION`. The primary classification (`is_blurry`) uses Laplacian variance only, so this is informational.

---

### BLUR-3 — Detect endpoint ignores user's config threshold (INFO) `ai-api`

**What:** The `/blur/detect` endpoint accepts a `threshold` query parameter (default 100.0). The `BlurDetector` instance is initialized with `settings.BLUR_THRESHOLD`. The endpoint passes the user's threshold via `threshold_override`, which correctly overrides per-request. However, the batch detect task does not accept or pass a custom threshold.

**Affected files:**
- `src/api/v1/blur.py:56` — single endpoint: `detector.detect(image)` — does NOT pass threshold_override
- `src/workers/tasks/blur_tasks.py:47` — batch: `detector.detect(image)` — does NOT pass threshold_override

**Wait — re-checking:** Actually, looking at `blur.py:56`, the single endpoint calls `detector.detect(image)` without the threshold override. The `threshold` query parameter is accepted but never used.

**Corrected finding:** The `/blur/detect` endpoint accepts a `threshold` query parameter but never passes it to `detector.detect()` as `threshold_override`. The parameter is dead code.

**Affected files:**
- `src/api/v1/blur.py:33` — `threshold: float = Query(default=100.0, ...)` — accepted
- `src/api/v1/blur.py:56` — `detector.detect(image)` — threshold not passed

**Impact:** Users who set a custom threshold see no effect. The instance default from config is always used.

---

### BLUR-4 — Classifier center-crop loses edge content (LOW) `ai-api`

**What:** `BlurClassifier._preprocess()` center-crops the image to a square before resizing to 224x224. For wide or tall images, significant edge content is cropped. If the blur defect is at the edges (e.g., motion blur at limbs in a running photo), it may be cropped out before inference.

**Affected files:**
- `src/ml/blur/classifier.py:119-123` — center-crop to square using shorter dimension

**Impact:** Classification accuracy degrades for images where blur is concentrated at the edges rather than the center. This is a known tradeoff of the YOLOv8-classify preprocessing pipeline.

---

### BLUR-5 — Class name ordering must match trained model exactly (MEDIUM) `ai-api`

**What:** The hardcoded class names in `BlurClassifier._load_model()` are alphabetically ordered: `["defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp"]`. If the ONNX export or training script produced a model with a different class order, all probabilities are silently misaligned. A `class_names.json` file can override defaults, but there is no runtime validation that the file matches the model's actual output layer ordering.

**Affected files:**
- `src/ml/blur/classifier.py:47-53` — hardcoded default order
- `src/ml/blur/classifier.py:86-90` — override from JSON file, no validation

**Impact:** If class order is wrong, every classification result maps to the wrong label. "Sharp" would be reported as "defocused_blurred" or vice versa. This is a silent, hard-to-detect failure.

---

### BLUR-6 — Batch detect/classify do not return image dimensions (INFO) `ai-api`

**What:** Single-image endpoints return `image_dimensions: (w, h)` in the response. Batch task results do not include image dimensions — the worker only stores inference results, not metadata about the input image.

**Affected files:**
- `src/workers/tasks/blur_tasks.py:48` — batch detect result: `{"index": i, **detection}` — detection dict has no dimensions
- `src/workers/tasks/blur_tasks.py:105,111` — batch classify same

**Impact:** Minor inconsistency between single and batch response schemas.

---

## 4. Cross-Cutting / Architecture Boundary

### ARCH-1 — No documentation of ai-api vs backend responsibility boundary (HIGH) `both`

**What:** There is no document defining what ai-api is responsible for versus what the backend handles. Specific ambiguities:

| Question | Unclear |
|----------|---------|
| Who owns participant/event data? | Backend presumably, but no API contract |
| Who matches bib numbers to participants? | Unknown — ai-api returns raw OCR |
| Who manages event galleries for face search? | Neither — no event concept exists in ai-api |
| Who presents results to end users? | Backend presumably |
| Who decides acceptable confidence thresholds? | ai-api returns raw confidence, no filtering |

**Impact:** Without a clear contract, both teams may assume the other handles a concern (bib matching, event isolation, confidence filtering), leading to gaps in production.

---

### ARCH-2 — No api_key_id scoping for bib or blur results retrieval (LOW) `ai-api`

**What:** Job results are retrieved via `GET /api/v1/jobs/{job_id}`. The job `id` is a UUID that is unguessable, but the retrieval endpoint does not verify that the requesting API key matches the `api_key_id` that created the job. Any valid API key can poll any job if it knows the UUID.

**Affected files:** (Would need to check the jobs endpoint — not in scope of this analysis but noted as an architecture observation.)

**Impact:** Low risk due to UUID unguessability, but violates defense-in-depth. A leaked job ID from logs or webhooks could expose another tenant's results.

---

### ARCH-3 — Webhook payloads do not include result data (INFO) `ai-api`

**What:** When a batch job completes, the webhook payload contains only `{"job_id": "...", "result_count": N}`. The actual results (bib numbers, face matches, blur classification) are not included. The backend must make a follow-up GET request to retrieve results.

**Affected files:**
- `src/workers/helpers.py:60-63` — `dispatch_webhook_sync` payload is `{"job_id": ..., "result_count": ...}`

**Impact:** Two round-trips (webhook notification + GET results) instead of one. This may be intentional for payload size reasons, but it means the webhook alone is not actionable.

---

## Summary Table

| ID | Feature | Severity | Owner | Description |
|----|---------|----------|-------|-------------|
| FA-1 | Face | CRITICAL | both | No event/gallery isolation — cross-event contamination |
| FA-2 | Face | MEDIUM | ai-api | No bulk enrollment endpoint |
| FA-3 | Face | LOW | ai-api | Batch worker uses per-face DB queries instead of batch |
| FA-4 | Face | INFO | ai-api | Compare uses only first face from each image |
| FA-5 | Face | LOW | ai-api | Batch search hardcodes top_k and threshold |
| FA-6 | Face | MEDIUM | ai-api | No duplicate embedding guard on enrollment |
| FA-7 | Face | MEDIUM | ai-api | No list-persons endpoint |
| BIB-1 | Bib | HIGH | both | No bib search/lookup endpoint — unclear ownership |
| BIB-2 | Bib | HIGH | ai-api | No confidence threshold on OCR output |
| BIB-3 | Bib | MEDIUM | ai-api | Single-digit bib numbers silently rejected |
| BIB-4 | Bib | LOW | ai-api | YOLO detection confidence hardcoded |
| BIB-5 | Bib | MEDIUM | ai-api | OCR character substitution (O/0, I/1) not handled |
| BIB-6 | Bib | MEDIUM | ai-api | Fallback OCR on full image produces false positives |
| BIB-7 | Bib | INFO | ai-api | Batch bib task missing api_key_id passthrough |
| BLUR-1 | Blur | MEDIUM | ai-api | Laplacian variance is image-size-dependent |
| BLUR-2 | Blur | LOW | ai-api | FFT mask unreliable near MIN_DIMENSION |
| BLUR-3 | Blur | MEDIUM | ai-api | Detect endpoint accepts threshold param but never uses it |
| BLUR-4 | Blur | LOW | ai-api | Center-crop loses edge blur information |
| BLUR-5 | Blur | MEDIUM | ai-api | Class name ordering not validated against model |
| BLUR-6 | Blur | INFO | ai-api | Batch results missing image dimensions |
| ARCH-1 | Cross | HIGH | both | No documented ai-api/backend responsibility boundary |
| ARCH-2 | Cross | LOW | ai-api | Job results not scoped by api_key_id on retrieval |
| ARCH-3 | Cross | INFO | ai-api | Webhooks don't include result data |
