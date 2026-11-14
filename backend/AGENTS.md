# AGENTS.md — Backend (Kotlin + Spring Boot)

**Status:** All phases shipped and hardened; all four roles locked. Recent milestones — checkout/payment hardening (V39: actor-scoped idempotency, duplicate-session expiry, signed guest capabilities, locked fulfillment, PayMongo refunds), Google sign-in (V38: `POST /auth/google` for the website GIS button + mobile Credential Manager), OTP reset (V37), async watermark (V36). **Live status and the controller / test counts are not pinned here** — they drift. Read vault `VAULT-INDEX.md` / `backend/index.md` for status, and run `.\gradlew.bat test` (unit, Docker-free) + `.\gradlew.bat integrationTest` (Docker) for the totals. Remaining work is gap-driven in vault `backend/tasks.md`.

The live route reference — every endpoint, its auth, and **which clients consume it** — is vault `backend/api-surface.md`. Read that before adding an endpoint or assuming one is missing.

## Role in the project

Spring Boot service. Public API for website and mobile. Owns users, events, participants, photos, marketplace. Runs face/bib inference for the clients through AWS Rekognition — clients never call `ai-api` directly.

See root `AGENTS.md` for monorepo-wide rules and `docs/project-vision.md` for product context.

## Stack

| Concern | Choice |
|---------|--------|
| Language | Kotlin 1.9.25 |
| JDK | 21 |
| Framework | Spring Boot 3.5.14 |
| Build | Gradle Kotlin DSL (`build.gradle.kts`) |
| Auth | JWT bearer (jjwt 0.12.6) — HS256 |
| Password hash | BCrypt (strength 12) |
| DB | PostgreSQL 16 (Docker locally) |
| Migrations | Flyway |
| Storage | S3-compatible object storage (Cloudflare R2 in prod, AWS S3 SDK; local disk in dev) |

## Folder structure

Layer-based top level, **domain subpackages inside `service/` and `dto/`** — the domain subpackage is the unit of ownership. Route truth lives in vault `backend/api-surface.md`; do not list endpoints here.

```
com.quickpitik
├── controller/          HTTP edge only: bind → @Valid → class-level @PreAuthorize → rateLimiter.acquireOrThrow → one service call.
│                        Named by audience: Me* · Admin* · Public* · plain (guest/public routes, webhooks).
├── service/<domain>/    ALL business rules. One domain owns each table (see Boundaries). Subpackages:
│                        auth · events · photos · photographer · orders · earnings · disbursement · admin · profile · cart · runner
│                        + adapters/helpers: ai · storage · email · image · ratelimit · reference. Nothing sits at service/ root.
├── repository/          Spring Data + JPQL/native: predicates, ordering, locks, conditional bulk UPDATEs, projections. Flat.
├── entity/              JPA state + tiny pure helpers (e.g. Photo.isSellable). Enums beside their entity. Flat.
├── dto/<domain>/        Typed wire shapes + `Entity.toDto()` extension functions in the DTO file. Never Map<String, Any?>.
├── config/              One @ConfigurationProperties class per `app.*` prefix + bean wiring (Security, Cors, RestClient, Async, WebSocket, OpenApi).
├── security/            JWT filter/provider, AuthPrincipal, HMAC webhook verifiers, raw-body + download-disposition filters.
├── websocket/           session registries, AFTER_COMMIT broadcasters, handshake interceptors.
├── common/              ApiResponse envelope + advice, pagination, ErrorCodes, IdempotencyKey, startup runners, ProductionSecretsGuard.
└── exception/           typed exceptions + GlobalExceptionHandler.
src/main/resources/db/migration/   Flyway, append-only.  src/test/kotlin/…/support/   Postgres base + integration tests.
```

Split a `service/<domain>` subpackage when it passes ~25 files (the 2026-05-05 "25 files in `service/`" trigger was answered 2026-09-11 — vault `backend/decisions.md`). No feature-first move: ownership is enforced by the rules below, not by folders.

## Boundaries (binding for every agent)

1. **Controllers do five things**: bind, `@Valid`, class-level `@PreAuthorize`, `rateLimiter.acquireOrThrow`, one service call. Never inject a repository; never read storage to feed logic (streaming an already-authorized key is fine); never branch on business state. Multipart forms bind to a `@Valid @ModelAttribute` DTO.
2. **One rule, one home.** A predicate needed twice becomes a service method or an entity helper. Copying a check "to be safe" is the bug.
3. **Table ownership — one writing service per table.** `transactions`, `payout_cycles`, `payout_status_history` → `service/earnings`. `orders`, `order_items`, `payments`, `download_grants` → `service/orders`. `photos.status` + watermark columns → conditional `UPDATE` only (watermark pipeline, admin flags). `photographer_settings.verification_status`: APPROVED is written by `service/photographer` (automatic, `PhotographerSettingsService.approveIfReady`); admin keeps reject and reset (column-disjoint, `@DynamicUpdate`); the rest of that row → `service/photographer`. Reading another domain's repository for hydration is fine; writing it is not.
4. **Transactions.** Class-level `@Transactional` for plain CRUD. A method that calls PayMongo, Resend, ai-api/Rekognition, S3/R2, or decodes images must not hold a DB transaction — use the `TransactionTemplate` split (`OrderService.create`, `PhotoUploadService.persistNew`). `readOnly = true` only on methods that can never reach `save()` or a lazy-create (vault `backend/notes/jpa-pitfalls.md`).
5. **Concurrent columns.** Rows the pipelines mutate (`photos`) change by conditional bulk `UPDATE … WHERE status = X`, never load-modify-`save()`. A multi-column row with concurrent writers carries `@DynamicUpdate`.
6. **Side effects.** Publish an `ApplicationEvent` inside the transaction; consume with `@TransactionalEventListener(AFTER_COMMIT)`; `@Async` on the pool for that work class (`imageProcessing`, `watermarkProcessing`, `emailSending`) — outage-prone I/O never on the indexing pool; every async path has a bounded `@Scheduled` sweep with a config'd interval and a semantic-vs-transport retry budget.
7. **Errors.** Throw `NotFound/Conflict/Unauthorized/ValidationException` or `ApiException(status, code)`; codes only from `ErrorCodes`; a provider failure on a user path is mapped, never left to the 500 catch-all; provider text never reaches the envelope.
8. **Config.** Every `app.*` key belongs to a `@ConfigurationProperties` class in `config/`; no `@Value` outside `config/` and `ProductionSecretsGuard`; dev defaults in `application.yml`; prod placeholders fail closed.
9. **Integrations.** Concrete `*Client` until a second implementation exists, then an interface named for the capability (`FaceBibProvider`, `StorageService`). Inject `RestClient` from `RestClientConfig`. Business limits live in properties + the calling service, not in the adapter.
10. **Authorization.** Role = class-level `@PreAuthorize`; ownership = query-scoped lookup (`findByIdAndUserId`…), 404 on miss; guest = capability token service; sensitive = `StepUpService.enforce` in the controller; never trust a client-supplied owner id. New public/guest routes get a row in `SecurityMatrixIntegrationTest`.
11. **Wire contract.** Typed DTOs only (webhook acks excepted); envelope is automatic; lists are `PaginatedResponse`; breaking changes go under `/api/v2`; update vault `api-surface.md` in the same commit.
12. **Time & money.** New time-window logic takes a `Clock`; one zone constant. `BigDecimal` scale 2; `toCentavos()`; charged amount = `pricePhpAtPurchase − discountPhp`, computed once.
13. **Repositories.** `@Query` carries predicates/ordering only; every new `@Query` gets a Postgres integration test.
14. **Tests.** Plain `Mockito.mock` + constructor injection, backticked names, mirrored packages; every new service method gets a unit test; `gradlew test` stays Docker-free; `integrationTest` is never skipped because Docker is down.
15. **Board.** Read vault `ACTIVE.md` first; a claimed lane's files are off-limits; claim your own.

## Module boundaries

- **Website + mobile → this backend only.** Never bypass to `ai-api`.
- **This backend → AWS Rekognition for face/bib** (`AI_PROVIDER=rekognition`, the default). The `ai-api` provider (`AiApiClient`) remains, but ai-api has had no face/bib routes since 2026-10-16.
- **Event isolation:** every face call is scoped to one event (Rekognition: one collection per event).
- **Confidence thresholds:** apply per-event here, not in the provider.

## Local development

### Prerequisites
- JDK 21 (already installed)
- Docker (for Postgres)

### First-time setup

```powershell
# 1. Start Postgres
cd backend
docker compose up -d postgres

# 2. Boot the app (Flyway runs V1 migration on first start)
.\gradlew.bat bootRun
```

App boots on `http://localhost:8080`. Frontend talks to `http://localhost:8080/api/v1`.

### Run tests

**Stop — read `docs/testing-cloud-services.md` before `integrationTest` in the main checkout.** When
`application-local.yml` exists, the suite also makes real AWS Rekognition calls with the local keys. The
R2 tests need `R2_SMOKE_BUCKET` (the test bucket); that file's own bucket is the production one.

```powershell
.\gradlew.bat test              # unit suite — no Docker, no Postgres
.\gradlew.bat integrationTest   # real Postgres via Testcontainers — needs Docker running
```

`test` runs the Mockito unit suite and **excludes** anything tagged `integration`, so it still
works on a machine with nothing but a JDK. `integrationTest` runs the other half against a
throwaway Postgres 16 container: Flyway V1→latest applying to a virgin database, `ddl-auto: validate`
proving the entities still match, the `uq_photos_photographer_content_hash` partial index, and the
lockout counter's survival of a rolled-back login transaction. Extend `PostgresIntegrationTest`
to add one. End-to-end is verified via `curl` (see Smoke Test below).

**If Docker is down when you need `integrationTest`, start it — never skip.** Bring Docker Desktop
up (launch it, then poll `docker info` until it succeeds), then run the suite. "Couldn't run —
Docker was down" is not a verified test; only report an integration test unrun if the daemon
genuinely won't start on this machine, and say so explicitly.

**Before a PR, run locally:** all of `test`, plus the integration classes your change touches or
adds (`.\gradlew.bat integrationTest --tests "*Payout*"`). The whole integration suite runs on the
PR's required `integration` check (`.github/workflows/backend-tests.yml`), so wait for that to go
green rather than running the whole suite on your machine. Run it locally only when the owner asks
or CI can't run it.

**One integration run per machine at a time.** Each run holds about 4 GB of memory plus its own
Postgres container, and two at once slow each other down badly. If `docker ps` shows a
`testcontainers/ryuk` container, another run is going: wait for it to finish.

### API docs

With the app running, `http://localhost:8080/swagger-ui.html` (spec at `/v3/api-docs`).
Generated from the controllers by springdoc; `OpenApiConfig` rewrites the response schemas into the
`{ success, data, errors }` envelope that `ResponseEnvelopeAdvice` actually emits, so what the page
shows is what the wire carries. The padlocks are declared globally — public routes show one they
don't enforce; `config/SecurityConfig.kt` is authoritative. Set `API_DOCS_ENABLED=false` to remove
the surface entirely.

Actuator (2026-08-27): `GET /actuator/health` is public; `/actuator/metrics` (+ `info`) require an
ADMIN bearer. `qp.*` metrics: `qp.upload.duration`, `qp.upload.validation.duration`,
`qp.upload.dedup{outcome}`, `qp.upload.direct{outcome}`, `qp.upload.cleanup{outcome}`,
`qp.upload.direct.pending.{count,bytes,oldest.age}`,
`qp.indexing.outcome{outcome,provider}`, `qp.ai.call{op}`, `qp.ratelimit.denied{policy}`,
`qp.watermark.cache{result}`, `qp.watermark.outcome{outcome}` (live/failed/transport, 2026-08-28),
`qp.removal.erasure{outcome}`, `qp.removal.erasure.open`, `qp.removal.evidence{outcome}`,
face search (2026-10-27): `qp.search.request{mode=oneshot|stored|library|alert}`, `qp.search.selfie.fetch`,
`qp.search.photo.query`, `qp.ai.error{op,code}`, `qp.indexing.reconciled`, `qp.indexing.released`, `qp.watermark.reconciled`,
`qp.relevance.lookups`, `qp.relevance.scored{level}`, `qp.relevance.tick`, `qp.relevance.check.tick`,
`qp.relevance.measured{result=ok|error|retry}`, `qp.relevance.measure` (one photo's read + label check) —
plus Hikari/JVM/HTTP-server metrics for free. Each face search also logs one INFO line
(`face search mode=… total_ms=… r2_ms=… aws_ms=… query_ms=… results=…`); metrics reset on every deploy.

### Wipe a test event

Run `backend/scripts/nuke-test-event-photos.sql` with your event IDs swapped in (for example `docker exec -i quickpitik-postgres psql -U quickpitik -d quickpitik < backend/scripts/nuke-test-event-photos.sql`). It deletes the dependent rows (orders, payments, downloads, disputes, transactions, photo_bibs, photo_face_persons) in dependency order. Then delete the photo files: `Remove-Item -Recurse -Force backend\.storage\events\<event-id>`.

### Delete an account on request

Runner (buyer) accounts: the owner runs `backend/scripts/delete-runner-account.sql` (a dry run unless `-v confirm=yes`). Its header comment is the procedure: what it deletes and keeps, the R2 clean-up, and how to open `psql` on production. Photographer accounts are never deleted (the sales ledger and payouts would cascade away): `backend/scripts/anonymise-photographer-account.sql` scrubs and suspends the account and keeps every money row; it refuses while unsold photos remain or money is still owed. Never use the `nuke-*` scripts for this: they delete money rows. The request log is the owner's private sheet, never in git or the vault.

### Stop Postgres

```powershell
docker compose down            # keeps data volume
docker compose down -v         # wipes data (forces V1 to re-run on next boot)
```

## Environment variables

All env vars have dev-friendly defaults in `application.yml` so the app boots out of the box. Override in production via OS env, `.env`, or a `application-prod.yml` (gitignored).

| Variable | Default | Production behavior |
|----------|---------|---------------------|
| `SPRING_PROFILES_ACTIVE` | *(none)* | **Set `prod` in production.** Activates `ProductionSecretsGuard`, which refuses to boot while any secret below is still its `dev-only` placeholder, the bootstrap password is the default, `STORAGE_BACKEND` is not `S3`, or `PAYMONGO_REFUND_LINK_ALLOWED_PREFIXES` is empty or holds an entry that is not an `https://` PayMongo address ending in `/` (host `paymongo.com` or `*.paymongo.com`, no user-info); also disables the dev mock-photo seeder. Railway runs from `backend/Dockerfile` (JDK 21, fontconfig, honours `PORT`); the variable list lives in vault `backend/notes/production-deploy.md`. |
| `DB_HOST` | `localhost` | RDS endpoint |
| `DB_PORT` | `5432` | — |
| `DB_NAME` | `quickpitik` | — |
| `DB_USER` | `quickpitik` | — |
| `DB_PASSWORD` | `quickpitik` | **MUST OVERRIDE** |
| `JWT_SECRET` | dev-only string (insecure, marked) | **MUST OVERRIDE** — generate with `openssl rand -base64 64` |
| `ORIGIN_SECRET` | *(blank)* | **MUST OVERRIDE** under `prod`; Cloudflare must overwrite `X-Origin-Secret` on every request to `api.quickpitik.com` with the same value. `/actuator/health` alone is exempt for Railway's direct health probe. Set the edge rule and Railway healthcheck before deploying the guarded backend. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated list of frontend origins |
| `ADMIN_BOOTSTRAP_EMAIL` | `admin@quickpitik.local` | Set per environment |
| `ADMIN_BOOTSTRAP_PASSWORD` | `changeme123` | **MUST OVERRIDE** |
| `ADMIN_BOOTSTRAP_NAME` | `QuickPitik Admin` | — |
| `SERVER_PORT` | `8080` | — |
| `GOOGLE_CLIENT_ID` | *(blank)* | Google OAuth **Web application** client ID — the shared audience for `/auth/google`. Blank makes the endpoint answer 503; the website (`NEXT_PUBLIC_GOOGLE_CLIENT_ID`) and mobile (`QP_GOOGLE_SERVER_CLIENT_ID`) carry the same value and hide their buttons when unset. |
| `AI_API_ENABLED` | `false` | Set `true` when ai-api is wired and you want face/bib indexing on upload + selfie quality gate + runner face-search. When `false`, every server-side ai-api call is skipped — photographer upload + selfie upload still work, face-search returns 503. See vault `backend/decisions.md` 2026-05-18 master-switch ADR. |
| `AI_PROVIDER` | `rekognition` | Which face/bib provider runs: `rekognition` (AWS; needs AWS keys or a role) or `ai-api`. ai-api lost its face/bib routes on 2026-10-16, so `ai-api` now gets 404s. Tests pin `ai-api` in `application-test.yml` so they keep their mocked client. `INDEXING_MODE=batch` is ai-api-only, so startup refuses it with `rekognition`. |
| `MODERATION_IMAGE_MIN_CONFIDENCE` | `0.8` | 0–1 floor for AWS DetectModerationLabels on photographer avatar, cover and watermark uploads (sent ×100). Not yet measured (AA-0.5). Raise it if real images are refused wrongly; a restart, no deploy. |
| `MODERATION_IMAGE_BLOCKED_LABELS` | `Explicit Nudity,Explicit,Non-Explicit Nudity of Intimate parts and Kissing,Violence,Visually Disturbing,Hate Symbols` | Comma list; an image is refused (`400 IMAGE_NOT_ALLOWED`) when a label's name or parent is listed. With `AI_API_ENABLED=false` or an AWS error the image is accepted and an admin flag (`profile_image_unchecked`) is raised. Needs IAM `rekognition:DetectModerationLabels`. |
| `SELF_SERVE_RACES_ENABLED` | `false` | Shared race events (auto-approval Plan 2) rollout switch: photographers create, join and edit races and set a race price. Off = today's flow. `GET /race-types` reports it as `selfServeEnabled`. |
| `RACE_DATE_WINDOW_PAST_DAYS` | `3` | A new race's date may be at most this many days ago, else it goes to admin review. Not yet measured. |
| `RACE_DATE_WINDOW_AHEAD_DAYS` | `90` | A new race's date may be at most this many days ahead, else admin review. Not yet measured. |
| `RACE_DAILY_CREATE_CAP` | `3` | Races one photographer creates per Manila day before the next goes to admin review. Not yet measured. |
| `RACE_DUP_STRONG_SCORE` | `0.85` | Duplicate-race score (0–1) at which a new race is offered as a join (tuned on the AA-R2 fixture). |
| `RACE_DUP_MEDIUM_SCORE` | `0.5` | Duplicate-race score (0–1) at which candidates are listed (tuned on the AA-R2 fixture). |
| `RACE_DUP_DATE_TOLERANCE_DAYS` | `1` | Days apart two races may be and still count as the same race. |
| `RACE_PROOF_MIN_PHOTOGRAPHERS` | `2` | Photographers with photos that prove a race is real when uploads close. |
| `RACE_PROOF_MIN_INDEXED_PHOTOS` | `20` | Indexed photos a race needs for the proof check. Not yet measured (AA-0.6). |
| `RACE_PROOF_MIN_BIB_SHARE` | `0.30` | Share (0–1) of indexed photos with a read bib that proves a race. Not yet measured (AA-0.6). |
| `RACE_PROOF_MIN_INDEXED_SHARE` | `0.5` | Below this share (0–1) of indexed photos the proof check only flags, never hides. Not yet measured. |
| `RACE_PROOF_SWEEP_INTERVAL_MS` | `3600000` | Cadence of the race proof sweep. |
| `RACE_PROOF_BIB_FLAG_ONLY_TYPES` | `CYCLING` | Comma list of race types whose failed bib rule only flags the race for an admin, never hides it (plates are rarely read). Not yet measured (AA-0.5). |
| `RACE_SAME_BIB_MIN_SHARED` | `30` | Bibs two races must share before they are flagged as one race. Not yet measured. |
| `RACE_REPORT_HIDE_THRESHOLD` | `3` | Distinct accounts whose reports hide an unproven race. Not yet measured. |
| `RACE_PRICE_CEILING_MULTIPLIER` | `2.0` | A photographer's own race price may go up to this times the race price (`PUT /me/photographer/events/{id}/price`, AA-R11). Owner decision D3. |
| `RACE_PRICE_COOLDOWN` | `PT24H` | How long after a race price change the photographer must wait before the next one. Not yet measured. |
| `RELEVANCE_MODE` | `off` | Photo relevance rollout and kill switch, read through `RelevanceSwitch`: `off` = nothing runs; `record-only` = relevance rows and admin alerts only; `auto-hide` = may hide (version 2), but runs as `record-only` with a boot WARN until `ADMIN_FLAGS_ENABLED` is on and both `RELEVANCE_JUNK_*_CUTOFF` are set. Old names still boot: `shadow` = `record-only`; `flag` = retired in RV-1.10 (P2): runs as `record-only` with a boot WARN and raises no flag; version 1 flags already open stay in the admin flags queue until closed. Today nothing hides, holds or deletes in any mode. Independent of `AI_API_ENABLED`. Owner sets it; an unknown value stops boot. |
| `RELEVANCE_SWEEP_INTERVAL_MS` | `60000` | Cadence of the relevance sweep; also its first delay after boot. |
| `RELEVANCE_QUIET_MINUTES` | `10` | A photographer's batch in an event is scored this long after their newest upload. |
| `RELEVANCE_LOOKUPS_PER_TICK` | `100` | Copy look-ups per sweep tick, all batches together (1–1000). |
| `RELEVANCE_MAX_BATCHES_PER_TICK` | `5` | Due batches read per sweep tick. |
| `RELEVANCE_RESCORE_MINUTES` | `60` | Wait before re-scoring a batch that was still processing or reading EXIF. |
| `RELEVANCE_SETTLE_DAYS` | `7` | After uploads close, stop waiting for late EXIF after this many days. |
| `RELEVANCE_LOOKUP_TIMEOUT` | `5s` | Statement timeout of one copy look-up transaction; a slower one fails that batch's step and the next tick tries again. |
| `RELEVANCE_COPY_TIGHT_DISTANCE` | `4` | pHash bits (0–64) within which an earlier photo by another photographer is a copy hit. Not yet measured (PE-0). |
| `RELEVANCE_COPY_WEAK_DISTANCE` | `8` | Upper end of the weak-evidence band (≥ tight distance). Not yet measured (PE-0). |
| `RELEVANCE_COPY_MIN_HITS` | `3` | Copy hits a batch needs before the copy rule can mark it Low. Not yet measured (PE-0). |
| `RELEVANCE_COPY_MIN_HIT_SHARE_PERCENT` | `5` | Percent (1–100) of a batch's look-ups that must be hits for Low. Not yet measured (PE-0). |
| `RELEVANCE_DATE_WINDOW_DAYS` | `2` | Days either side of the event date a capture date may fall and still count as inside. Not yet measured (PE-0). |
| `RELEVANCE_DATE_MIN_DATED_PHOTOS` | `10` | Photos with a capture time a batch needs before the date rule can mark it Low. Not yet measured (PE-0). |
| `RELEVANCE_DATE_OUTSIDE_SHARE_PERCENT` | `80` | Percent (1–100) of dated photos outside the window for Low. Not yet measured (PE-0). |
| `RELEVANCE_JUNK_BLUR_CUTOFF` | unset | Version 2: foreground sharpness score (0–100) below which a photo counts as blurry. Unset until the calibration run (O7); while unset, `auto-hide` runs as `record-only`. |
| `RELEVANCE_JUNK_DARK_CUTOFF` | unset | Version 2: brightness score (0–100) below which a photo counts as dark. Unset until the calibration run (O7); while unset, `auto-hide` runs as `record-only`. |
| `RELEVANCE_JUNK_LABEL_MIN_CONFIDENCE` | `0.6` | Version 2: a photo is "not a race" when neither the `Person` label nor any race label reaches this confidence (0–1). Not yet measured (O7). |
| `RELEVANCE_JUNK_RACE_LABELS` | `Running,Marathon,Road` | Version 2: the image labels that count as a race scene (comma-separated). Not yet measured (O7). |
| `RELEVANCE_JUNK_FITS_BELOW_PERCENT` | `10` | Version 2: a measured sample with a junk share under this percent is in the "under" band (may be "fits"). |
| `RELEVANCE_JUNK_DUMP_ABOVE_PERCENT` | `70` | Version 2: a junk share over this percent is in the "over" band (may be "junk dump"); between the two lines is "mixed". Below 100, at least the fits line. |
| `RELEVANCE_CHECK_MIN_BATCH_SIZE` | `30` | Version 2: a batch with fewer photos is "small" and can only be "fits" or "can't tell". |
| `RELEVANCE_CHECK_NEAR_LINE_POINTS` | `5` | Version 2: a junk share within this many points of the dump line gets a second sample before the verdict. |
| `RELEVANCE_CHECK_SAMPLE_SIZE` | `300` | Version 2: photos picked for one sample round of a check; a smaller batch is sampled whole. |
| `RELEVANCE_CHECK_COUNT_TRIGGER` | `500` | Version 2: a batch still uploading is checked again once it has grown by this many photos since its last check (a quiet batch is checked `RELEVANCE_QUIET_MINUTES` after its newest upload). |
| `RELEVANCE_CHECK_TICK_INTERVAL_MS` | `15000` | Version 2: cadence of the check tick; also its first delay after boot. With `AI_API_ENABLED` off no check starts. |
| `RELEVANCE_CHECK_MAX_BATCHES_PER_TICK` | `3` | Version 2: due batches a check tick starts at most. Public events only. |
| `RELEVANCE_CHECK_CLAIM_MINUTES` | `10` | Version 2: how long one server copy's claim on a batch holds before another copy may take it. |
| `RELEVANCE_CHECK_MEASURES_PER_TICK` | `60` | Version 2: sampled photos one check tick measures at most (one AWS DetectLabels call each). A photo measured by an earlier check is not measured again. Needs `AI_API_ENABLED` on and the Rekognition provider. |
| `RELEVANCE_CHECK_WORKERS` | `2` | Version 2: threads of the `relevanceScan` pool that read and measure sampled photos, apart from the scheduler and the upload pools. |
| `RELEVANCE_CHECK_MEASURE_MAX_TRIES` | `5` | Version 2: a sampled photo that hits a throttle, server error or timeout this many times is given up (`GAVE_UP`) and left out of the junk share; it is never counted as junk. |
| `RELEVANCE_CHECK_WAIT_FOR_RULES_MINUTES` | `30` | Version 2: a measured check waits for version 1 to score the same upload, at most this long; then it is decided with the copy and date results as they stand (`not_finished` / `not_run`). |
| `RELEVANCE_CHECK_DECIDE_MAX_ATTEMPTS` | `3` | Version 2: model calls per check, one per tick; when the last one fails the check is stored as can't tell (`CALL_FAILED`). Each call has its own `RELEVANCE_LLM_MAX_ATTEMPTS` tries. |
| `RELEVANCE_WORDS_MIN_SHARE_PERCENT` | `3` | Version 2: words read off a batch's photos (non-bib text, letters only, kept as per-batch counts while the mode is not `off`) are deleted once the batch has settled (`RELEVANCE_SETTLE_DAYS` after its final scoring) when they are on under this share of its photos. 1 to 100. |
| `RELEVANCE_WORDS_TOP_N` | `30` | Version 2: at most this many of a batch's words, each on at least `RELEVANCE_WORDS_MIN_SHARE_PERCENT` of its photos, go into the summary sent to the model; the same words are counted across the event's other photographers. 1 to 100. |
| `RELEVANCE_LLM_API_KEY` | blank | Version 2: the key for the Claude Messages API that gives each check its verdict. Secret, never logged. Blank = no model call is ever made and no check is decided (`RelevanceSwitch.canDecide` is false). The owner creates and sets it. Deliberately not `ANTHROPIC_API_KEY`, which many developer shells already hold. |
| `RELEVANCE_LLM_BASE_URL` | `https://api.anthropic.com` | Version 2: root of the Claude API (`POST /v1/messages`, API version `2026-10-21`). |
| `RELEVANCE_LLM_MODEL` | `claude-sonnet-5-5` | Version 2: the model that gives the verdict (O6). |
| `RELEVANCE_LLM_CONNECT_TIMEOUT` / `RELEVANCE_LLM_READ_TIMEOUT` | `5s` / `30s` | Version 2: limits of one model call; a timeout counts as a retryable failure. |
| `RELEVANCE_LLM_MAX_ATTEMPTS` | `3` | Version 2: tries per model call, the first included (1 to 5). A 429, a 5xx or a timeout is retried; any other error, a refusal or a cut-off answer is not. |
| `RELEVANCE_LLM_RETRY_BACKOFF` | `1s` | Version 2: wait before each retry of a model call. |
| `RELEVANCE_LLM_MAX_TOKENS` | `4096` | Version 2: `max_tokens` of one model call, the model's thinking included; an answer cut off at it is stored as can't tell (`CUT_OFF`). |
| `RELEVANCE_ALERTS_BIG_HIDE_PHOTOS` | `500` | Version 2: a hide of at least this many photos in one batch raises one stored admin alert (in record-only, the hide that would have happened: the whole batch, or the sample's junk share of it). |
| `RELEVANCE_HIDE_CHUNK_SIZE` | `1000` | Version 2 auto-hide: photos hidden per chunk, each chunk its own short transaction (1 to 1000; no statement binds more ids). |
| `RELEVANCE_HIDE_CHUNKS_PER_TICK` | `10` | Version 2 auto-hide: chunks per case in one check tick; the rest of a batch is hidden on the next ticks. |
| `IDENTITY_GATE_ENABLED` | `false` | Off = today's upload rule (any seller details not REJECTED). On = new uploads need seller details AUTO_APPROVED or APPROVED; MANUAL_REVIEW is refused too, still `403 SELLER_DETAILS_REQUIRED`; photos already reserved still finish (auto-approval AA-I6). |
| `IDENTITY_GATE_STARTS_ON` | blank | The announced date the identity gate turns on (`YYYY-MM-DD`), sent as `identityGateStartsOn` on the photographer's seller-details responses, and the date the identity notice email names (`POST /admin/identity-notice` refuses while it is blank). Blank = none; an invalid date fails boot. |
| `IDENTITY_MAX_FAILED_AUTO_CHECKS` | `3` | Failed automatic identity tries (a check the person failed: liveness, face, no face on the ID, ID number or legal name; never an AWS error or AI off) after which every later try, even a pass, goes to an admin with `REPEATED_FAILED_CHECKS`. Only an admin approval of the seller details resets the count to 0; admins see it as `failedAutoChecks`. At least 1 (AA review R-8). |
| `IDENTITY_NOTICE_SEND_PER_SECOND` | `2` | How many identity-notice mails `POST /admin/identity-notice` sends a second (one at a time; 1-1000). Keep it under Resend's rate limit [NOT MEASURED] (auto-approval AA-I22). |
| `LIVENESS_REGION` | `ap-southeast-5` | Region for AWS Face Liveness (session create, result read, the browser's STS keys). Needs IAM `rekognition:CreateFaceLivenessSession` + `GetFaceLivenessSessionResults` there. The website's security policy allows only `wss://streaming-rekognition.ap-southeast-5.amazonaws.com`: a new region means changing `website/next.config.ts`, the browser role's region, and redeploying the website. |
| `LIVENESS_MIN_CONFIDENCE` | `95` | 0–100 liveness score an automatic identity pass needs. Strict until measured (AA-0.4). |
| `FACE_MIN_SIMILARITY` | `99` | 0–100 CompareFaces similarity (live selfie vs ID front) an automatic pass needs. Strict until measured (AA-0.4). Not the runner face-search threshold. |
| `LIVENESS_ATTEMPTS_PER_DAY` | `3` | Live-selfie sessions per account per Manila day; the 4th answers `429 LIVENESS_ATTEMPTS_EXHAUSTED` with `Retry-After` to the next Manila midnight. |
| `LIVENESS_ROLE_ARN` | *(blank)* | ARN of IAM role `quickpitik-liveness-browser` (AA-0.2), assumed once per session for the browser's keys. Blank, `AI_API_ENABLED=false` or `AI_PROVIDER` not `rekognition` → `503 LIVENESS_UNAVAILABLE`. Not a secret, but keep the account id out of the vault. |
| `ID_NUMBER_HMAC_SECRET` | dev-only placeholder | **MUST OVERRIDE**, a different value per environment. The typed government ID number is stored only as HMAC-SHA256(this, `<ID_TYPE>:<number>`) (AA-I4). Never rotate it casually: older rows stop matching repeat accounts (AA-I5). |
| `ID_NUMBER_FORMAT_<TYPE>` | *(blank)* | One per ID type (`PHILSYS`, `DRIVERS_LICENSE`, `PASSPORT`, `UMID`, `PRC`, `POSTAL`, `VOTERS`): a regex the normalised number (uppercase, no spaces or dashes) must match, else `400 VALIDATION_ERROR` on `idNumber`. Blank = an admin checks this ID type: a number found on the card still ends in manual review (`ID_FORMAT_UNCHECKED`), never an automatic approval. An invalid regex fails boot. Set from AA-0.5. |
| `LIVENESS_CREDENTIAL_SECONDS` | `900` | Life of the browser's keys (900–3600 s; STS minimum 900). |
| `AUTH_LOCKOUT_MAX_ATTEMPTS` | `5` | Consecutive failed logins before an account locks (V29). Raise it if a demo needs more headroom — there is no separate on/off switch. |
| `AUTH_LOCKOUT_DURATION` | `PT15M` | How long a lock holds. Auto-clears; a successful login also resets it. Unlike the rate-limit buckets this is **always on**, independent of `RATE_LIMIT_ENABLED`. |
| `AUTH_LOCKOUT_WINDOW` | `PT15M` | NFR-S-14 (V34): failures only count toward a lock when within this window of the previous one; older streaks restart at 1. |
| `RATE_LIMIT_ENABLED` | `true` | Token buckets (bucket4j, in-memory). NFR-S-11 windows: auth 10/15 min per IP, photo-search 30/15 min; plus order-create, bundle-download, media-upload, photographer-upload, public-gallery policies. Set `false` only for load tests. Per-IP keys use `remoteAddr` — behind a proxy configure `server.tomcat.remoteip.*`. |
| `ORDER_CAPABILITY_SECRET` | dev-only placeholder | **MUST OVERRIDE** with at least 32 random bytes; signs purpose-bound guest return and bundle links. Keep stable or outstanding links become invalid. |
| `PAYMONGO_CHECKOUT_TTL` | `PT30M` | Age after which the reconciler expires an unpaid Checkout Session and releases its photos for a fresh checkout. |
| `COUPON_MAX_PERCENT` | `50` | Largest photographer coupon (V45), as a whole percent of the photographer's share. A coupon only lowers what the photographer keeps — the platform cut on a sale never moves, whatever this is set to. One exception (2026-09-05): exactly **100%** on a paid event the photographer *created* is a free giveaway — list price waived, platform cut included — and a checkout whose every order totals ₱0 settles without PayMongo (`PaymongoWebhookService.settleFree`). |
| `AI_MAX_INDEXING_ATTEMPTS` | `5` | Per-photo indexing retry budget. Only *semantic* failures (bad image, 4xx) consume it — transport failures (provider unreachable) return the photo to PENDING with the budget intact. `POST /admin/events/{id}/photos/reindex` re-drives exhausted/stale photos (`?all=true` after a provider flip). |
| `AI_RECONCILE_INTERVAL_MS` | `60000` | Cadence of the per-photo indexing reconcile sweep. |
| `WATERMARK_MAX_ATTEMPTS` | `5` | Async-watermark retry budget per photo (V36). Only semantic failures (undecodable bytes) consume it; transport failures leave it intact so the sweep keeps re-driving. Exhausted photos stay `PROCESSING` (owner-visible, runner-invisible) — watch `qp.watermark.outcome{outcome=failed}`. |
| `WATERMARK_RECONCILE_INTERVAL_MS` | `60000` | Cadence of the watermark reconcile sweep (re-drives stuck `PROCESSING` photos) and of the V42 pHash backfill. |
| `WATERMARK_VERIFY_MAX_DISTANCE` | `12` | Max Hamming distance (of 64 pHash bits) for `POST /public/photos/verify` to report a match; ≤ 6 reports `strong`, 7–12 `weak`. Compared against the marked, clean and centre-crop hashes (V43). |
| `WATERMARK_SEED_SECRET` | dev-only placeholder | **MUST OVERRIDE.** Seeds each preview's stripe geometry via HMAC(secret, photoId). Keep stable: a change re-seeds every future render; existing previews are unaffected. |
| `PHOTO_REMOVAL_IN_REQUEST_CLEANUP_JOBS` | `5` | Photo-removal erasures a decision or cleanup retry attempts before answering (0–100); the cleanup sweep finishes the rest. Keeps a large authorization inside the proxy timeout. |
| `RUNNER_REMOVAL_LIVENESS_PASS` | `90` | Runner deletion request: live-check confidence (AWS 0-100) at or above which the check passes. Must be at least the uncertain value and at most 100. |
| `RUNNER_REMOVAL_LIVENESS_UNCERTAIN` | `50` | Confidence from here up to the pass value is held for an admin (nothing auto-hides); below it the check fails. |
| `RUNNER_REMOVAL_AUTO_HIDE_SIMILARITY` | `0.99` | Face-match similarity (0-1) at or above which a runner's match hides at once; must be at least the provider floor `app.ai.rekognition.face-match-threshold`. Lower matches are suggestions. |
| `RUNNER_REMOVAL_AUTO_HIDE_MAX` | `30` | Most photos one runner request may auto-hide; more holds them all as suggestions. `0` stops all auto-hides (kill switch). |
| `RUNNER_REMOVAL_PHOTOGRAPHER_DAILY` | `50` | Most auto-hides per photographer per event per day across runner requests; past it that photographer's matches become suggestions. |
| `RUNNER_REMOVAL_CANDIDATE_LIMIT` | `50` | Most face matches a runner request asks the provider for (1-4096). |
| `RUNNER_REMOVAL_EMAIL_CLAIM_STALE_AFTER` | `PT10M` | How long an email send claim may sit before the runner/photographer email sweeps retry it. |
| `PHOTO_REMOVAL_EMAIL_SWEEP_INTERVAL` | `PT15M` | How often the sweeps re-send photographer and runner removal emails whose send claim went stale and was never delivered (last 24 hours only). |
| `RUNNER_REMOVAL_CHECKS_PER_DAY` | `3` | Live checks one account may start per Manila day (shared with the seller check). `0` closes runner deletion intake (kill switch). |
| `RUNNER_REMOVAL_MAX_OPEN` | `3` | Open deletion requests one runner may have at once (1 or more). |
| `DISPUTE_REVIEW_BUSINESS_DAYS` | `3` | Business days (Mon-Fri, Manila; holidays not skipped) an admin has to review an open or escalated dispute; drives its due time and overdue flag. |
| `DISPUTE_REFUND_SEND_WITHIN` | `PT48H` | After a QR Ph refund is approved, how long the admin has to send the PayMongo refund link (earlier if PayMongo's refund window closes first). |
| `DISPUTE_LINK_EXPIRING_SOON` | `PT24H` | A sent refund link is flagged "expiring soon" this long before it stops working. |
| `DISPUTE_REFUND_STUCK_AFTER` | `PT24H` | An in-flight PayMongo refund older than this becomes a refund problem the admin must check. |
| `DB_POOL_SIZE` | `10` | Hikari `maximum-pool-size`. |
| `LOG_LEVEL_APP` | `INFO` | `com.quickpitik` log level (set `DEBUG` for local debugging). |
| `API_DOCS_ENABLED` | `true` | Serves `/swagger-ui.html` + `/v3/api-docs`. **Set `false` in production** — a full route inventory is free reconnaissance. |

`BootstrapAdminRunner` creates the admin on first boot if no `ADMIN` exists yet. Subsequent boots are no-ops.

## Smoke test (Phase A)

With Postgres up + app running:

```bash
# 1. Register a runner
curl -X POST http://localhost:8080/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Runner","email":"runner@test.local","password":"marathon-cebu-2026","role":"RUNNER"}'

# 2. Login
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"runner@test.local","password":"marathon-cebu-2026"}'

# 3. /me with bearer
curl http://localhost:8080/api/v1/auth/me \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

# 4. Refresh
curl -X POST http://localhost:8080/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refreshToken":"<REFRESH_TOKEN>"}'

# 5. Forgot password — check console for the [EMAIL STUB] 6-digit code
curl -X POST http://localhost:8080/api/v1/auth/forgot-password \
  -H "Content-Type: application/json" \
  -d '{"email":"runner@test.local"}'

# 6a. Verify the logged code — the response carries the one-shot resetToken
curl -X POST http://localhost:8080/api/v1/auth/verify-reset-otp \
  -H "Content-Type: application/json" \
  -d '{"email":"runner@test.local","code":"<CODE_FROM_LOG>"}'

# 6b. Reset with the continuation token from 6a
curl -X POST http://localhost:8080/api/v1/auth/reset-password \
  -H "Content-Type: application/json" \
  -d '{"token":"<RESET_TOKEN_FROM_6A>","newPassword":"newpassword123"}'

# 7. Login as the bootstrap admin
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@quickpitik.local","password":"changeme123"}'
```

All responses come wrapped in `{ success, data, errors? }`. Match the website `ApiResponse<T>` envelope exactly.

**Password gate:** register, reset-password, and change-password all run `PasswordValidator` (NIST SP 800-63B style — no composition rules, screened against a known-weak list). Guessable strings that clear the 8-char floor — `password123`, `12345678`, `changeme123` — 400 with `WEAK_PASSWORD`. Keep the passwords in these snippets off that list. Login is **not** screened, so step 7's bootstrap admin still works with its default.

## Auth contract (locked, matches website)

| Endpoint | Method | Auth | Body |
|----------|--------|------|------|
| `/api/v1/auth/register` | POST | Public | `{ name, email, password, role: "RUNNER"\|"PHOTOGRAPHER", termsAccepted?: true }` — `termsAccepted` stamps `users.terms_version` (`app.legal.current-version`) + `terms_accepted_at` (V91); optional, old builds omit it |
| `/api/v1/auth/login` | POST | Public | `{ email, password }` |
| `/api/v1/auth/google` | POST | Public | `{ idToken, role?: "RUNNER"\|"PHOTOGRAPHER", termsAccepted?: true }` — Google ID-token exchange; answers **422 `ROLE_REQUIRED`** for a brand-new Google account until `role` is supplied; `termsAccepted` counts only on the call that creates the account |
| `/api/v1/auth/refresh` | POST | Public | `{ refreshToken }` |
| `/api/v1/auth/forgot-password` | POST | Public | `{ email }` — mails a 6-digit OTP |
| `/api/v1/auth/verify-reset-otp` | POST | Public | `{ email, code }` → `{ resetToken }` |
| `/api/v1/auth/reset-password` | POST | Public | `{ token, newPassword }` — token comes from verify-reset-otp, never from the email |
| `/api/v1/auth/verify-email` | POST | Public | `{ token }` |
| `/api/v1/auth/resend-verification` | POST | Bearer JWT | — |
| `/api/v1/auth/me` | GET | Bearer JWT | — |

`AuthResponse`: `{ accessToken, refreshToken, user: { id, email, name, role, avatarUrl?, emailVerified, createdAt, termsVersion?, legalCurrentVersion, termsAcceptanceRequired } }`.

**Email verification is advisory** (V30). Registering mails a link (`AFTER_COMMIT`, `@Async`) and
stamps `users.email_verified_at` when redeemed, but **nothing gates on it** — both clients sign the
user in the moment `/auth/register` returns, so enforcing it is a cross-module decision. In dev the
link is logged as `[EMAIL STUB] verification for … — link: …` rather than sent.

**Login can answer `429 ACCOUNT_LOCKED`** with `Retry-After` after `AUTH_LOCKOUT_MAX_ATTEMPTS`
consecutive failures. To clear one by hand:
`UPDATE users SET locked_until = NULL, failed_login_attempts = 0 WHERE email = '…';`

Roles are UPPERCASE in JSON: `"ADMIN" | "PHOTOGRAPHER" | "RUNNER"`. `ADMIN` is not creatable via `/auth/register` — only via `BootstrapAdminRunner` on first boot, or admin-promotion endpoints (Phase G).

## Implementation notes

- **Google sign-in** (V38, 2026-08-29): `POST /auth/google` verifies a Google ID token against Google's JWKS via `spring-security-oauth2-jose` (`GOOGLE_CLIENT_ID` is the required audience; blank → 503 `GOOGLE_AUTH_UNAVAILABLE`, JWKS outage → 503, bad token → 401 `INVALID_GOOGLE_TOKEN`). `google_sub` match → sign-in. Email match → **auto-link with guard**: an account that never verified its email gets its password rotated to an unusable hash and every refresh token revoked before linking (pre-registration squatters keep nothing; the inbox owner OTP-resets a new password). Brand-new email → 422 `ROLE_REQUIRED`; the client re-POSTs the same token with a role and the account is created with an unusable random password, `email_verified_at` stamped, and **no verification mail**. Same rate policy as login. See `GoogleAuthService`.
- **Face templates expire** (V92, legal review M2): `EventFaceExpiry` deletes an event's Rekognition collection `app.ai-api.face-retention` (6 months) after the event date, or when the event is deleted, and stamps `events.faces_deleted_at` (a failed delete stays empty and the daily sweep retries; skipped when `AI_API_ENABLED=false`). After that, face search on the event answers **410 `FACE_SEARCH_ENDED`** and indexing reads bibs only. Bib search is unchanged. An event's date cannot be moved to or from a date past retention (400 `VALIDATION_ERROR` on `date`, photographer and admin edits).
- **Refresh tokens** are opaque random 32-byte base64url strings, hashed with SHA-256 before persistence. **Rotated on every refresh** (parent token revoked, new one issued). On `confirmReset`, all of a user's refresh tokens are revoked to log out other sessions.
- **Password reset is a 6-digit OTP** (V37): forgot-password mails a code (10-min TTL, SHA-256-hashed at rest, only the newest outstanding code is live, dead after 5 wrong verify attempts); verify-reset-otp trades it for a 15-min one-shot continuation token (opaque 32-byte, same generation as refresh tokens) that reset-password consumes. A row's `token_hash` is NULL until the code is verified, so the verify step cannot be bypassed. Verify fails identically for unknown email and wrong code (anti-enumeration).
- **Access tokens** are JWTs (HS256) carrying `sub=userId`, `email`, `role` claims. 15-min TTL.
- **Forgot-password** is intentionally silent if the email doesn't exist (anti-enumeration). The endpoint always returns the same generic message.
- **Email is real (Resend)**: `EmailService` sends via `service/email/ResendClient` (`RESEND_API_KEY`; a dev placeholder key is detected and logged instead of sent). Covers password reset, order receipts, change-email (V28), and advisory verification (V30) mails. Every mailed link's origin is the **first** `CORS_ALLOWED_ORIGINS` entry. (The stdout-stub description here was stale — the rewrite landed 2026-05-20; corrected 2026-08-19.)
- **Response envelope**: `ResponseEnvelopeAdvice` (in `common/`) wraps every controller return value in `ApiResponse.success(...)`. `GlobalExceptionHandler` returns `ResponseEntity<ApiResponse<Nothing>>` directly (already wrapped) so the advice doesn't double-wrap.

## Working notes live in Obsidian

See vault `QuickPitik Vault/backend/`:
- `index.md` — module overview
- `tasks.md` — current todos
- `decisions.md` — ADRs (folder structure, Phase A scope, deferral)
- `notes/folder-structure.md` — stub; the live structure + boundary rules are in this file (above)

## Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `WeakKeyException` on boot | `JWT_SECRET` shorter than 32 bytes | Regenerate longer secret |
| `FlywayException: Validate failed` after schema edit | A migration was modified after being applied | Drop volume (`docker compose down -v`) for dev, or write a new V2 |
| `org.postgresql.util.PSQLException: Connection refused` | Postgres not started | `docker compose up -d postgres` |
| `validate ddl-auto` errors on entity rename | Entity ↔ migration mismatch | Either change migration (dev) or write a new one (prod-shaped) |
| 401 on every authenticated request | `Authorization: Bearer ...` header missing or expired | Re-login or refresh |
| Previews show the QuickPitik wordmark tiles but no credit text | JDK image lacks fontconfig/freetype, so `WatermarkService` disabled text (one WARN at first render) | `apt-get install -y fontconfig` in the runtime image (`backend/Dockerfile` already does) |
| Boot fails with `Refusing to start with profile 'prod'` | A secret is still its dev placeholder (see `SPRING_PROFILES_ACTIVE` above) | Set the named variables; never unset the profile to get past it |
