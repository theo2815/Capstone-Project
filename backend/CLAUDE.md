# CLAUDE.md — Backend (Kotlin + Spring Boot)

**Status:** All phases shipped and hardened; all four roles locked. Recent milestones — checkout/payment hardening (V39: actor-scoped idempotency, duplicate-session expiry, signed guest capabilities, locked fulfillment, PayMongo refunds), Google sign-in (V38: `POST /auth/google` for the website GIS button + mobile Credential Manager), OTP reset (V37), async watermark (V36). **Live status and the controller / test counts are not pinned here** — they drift. Read vault `VAULT-INDEX.md` / `backend/index.md` for status, and run `.\gradlew.bat test` (unit, Docker-free) + `.\gradlew.bat integrationTest` (Docker) for the totals. Remaining work is gap-driven in vault `backend/tasks.md`.

The live route reference — every endpoint, its auth, and **which clients consume it** — is vault `backend/api-surface.md`. Read that before adding an endpoint or assuming one is missing.

## Role in the project

Spring Boot service. Public API for website and mobile. Owns users, events, participants, photos, marketplace. Proxies all `ai-api` calls — clients never call `ai-api` directly.

See root `CLAUDE.md` for monorepo-wide rules and `docs/project-vision.md` for product context.

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
| Storage | AWS S3 (Phase C onward) |

## Folder structure

Layer-based packaging under `com.quickpitik`:

```
entity/         — JPA entities (User, Role enum, RefreshToken, PasswordResetToken)
repository/     — Spring Data JPA interfaces
dto/auth/       — Request/response DTOs (Bean Validation on inbound)
service/        — Business logic (Auth, RefreshToken, PasswordReset, Email)
controller/     — HTTP endpoints (AuthController)
config/         — Spring @Configuration (Security, Cors, JwtProperties)
security/       — JWT plumbing (TokenProvider, Filter, UserDetailsService, AuthPrincipal, OpaqueTokens)
exception/      — Custom exceptions + GlobalExceptionHandler (@RestControllerAdvice)
common/         — Cross-cutting (ApiResponse envelope, ResponseEnvelopeAdvice, BootstrapAdminRunner)
```

Full structure (including future Phase B–G): vault `backend/notes/folder-structure.md`.

## Module boundaries

- **Website + mobile → this backend only.** Never bypass to `ai-api`.
- **This backend → ai-api via server-to-server with API key** (added Phase C).
- **Event isolation:** every `ai-api` face call MUST include `event_id` (Phase C).
- **Confidence thresholds:** apply per-event here, not in `ai-api`.

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

### API docs

With the app running, `http://localhost:8080/swagger-ui.html` (spec at `/v3/api-docs`).
Generated from the controllers by springdoc; `OpenApiConfig` rewrites the response schemas into the
`{ success, data, errors }` envelope that `ResponseEnvelopeAdvice` actually emits, so what the page
shows is what the wire carries. The padlocks are declared globally — public routes show one they
don't enforce; `config/SecurityConfig.kt` is authoritative. Set `API_DOCS_ENABLED=false` to remove
the surface entirely.

Actuator (2026-08-27): `GET /actuator/health` is public; `/actuator/metrics` (+ `info`) require an
ADMIN bearer. `qp.*` metrics: `qp.upload.duration`, `qp.upload.dedup{outcome}`,
`qp.indexing.outcome{outcome,provider}`, `qp.ai.call{op}`, `qp.ratelimit.denied{policy}`,
`qp.watermark.cache{result}`, `qp.watermark.outcome{outcome}` (live/failed/transport, 2026-08-28) —
plus Hikari/JVM/HTTP-server metrics for free.

### Stop Postgres

```powershell
docker compose down            # keeps data volume
docker compose down -v         # wipes data (forces V1 to re-run on next boot)
```

## Environment variables

All env vars have dev-friendly defaults in `application.yml` so the app boots out of the box. Override in production via OS env, `.env`, or a `application-prod.yml` (gitignored).

| Variable | Default | Production behavior |
|----------|---------|---------------------|
| `SPRING_PROFILES_ACTIVE` | *(none)* | **Set `prod` in production.** Activates `ProductionSecretsGuard`, which refuses to boot while any secret below is still its `dev-only` placeholder, the bootstrap password is the default, or `STORAGE_BACKEND` is not `S3`; also disables the dev mock-photo seeder. Railway runs from `backend/Dockerfile` (JDK 21, fontconfig, honours `PORT`); the variable list lives in vault `backend/notes/production-deploy.md`. |
| `DB_HOST` | `localhost` | RDS endpoint |
| `DB_PORT` | `5432` | — |
| `DB_NAME` | `quickpitik` | — |
| `DB_USER` | `quickpitik` | — |
| `DB_PASSWORD` | `quickpitik` | **MUST OVERRIDE** |
| `JWT_SECRET` | dev-only string (insecure, marked) | **MUST OVERRIDE** — generate with `openssl rand -base64 64` |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated list of frontend origins |
| `ADMIN_BOOTSTRAP_EMAIL` | `admin@quickpitik.local` | Set per environment |
| `ADMIN_BOOTSTRAP_PASSWORD` | `changeme123` | **MUST OVERRIDE** |
| `ADMIN_BOOTSTRAP_NAME` | `QuickPitik Admin` | — |
| `SERVER_PORT` | `8080` | — |
| `GOOGLE_CLIENT_ID` | *(blank)* | Google OAuth **Web application** client ID — the shared audience for `/auth/google`. Blank makes the endpoint answer 503; the website (`NEXT_PUBLIC_GOOGLE_CLIENT_ID`) and mobile (`QP_GOOGLE_SERVER_CLIENT_ID`) carry the same value and hide their buttons when unset. |
| `AI_API_ENABLED` | `false` | Set `true` when ai-api is wired and you want face/bib indexing on upload + selfie quality gate + runner face-search. When `false`, every server-side ai-api call is skipped — photographer upload + selfie upload still work, face-search returns 503. See vault `backend/decisions.md` 2026-05-18 master-switch ADR. |
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
| `/api/v1/auth/register` | POST | Public | `{ name, email, password, role: "RUNNER"\|"PHOTOGRAPHER" }` |
| `/api/v1/auth/login` | POST | Public | `{ email, password }` |
| `/api/v1/auth/google` | POST | Public | `{ idToken, role?: "RUNNER"\|"PHOTOGRAPHER" }` — Google ID-token exchange; answers **422 `ROLE_REQUIRED`** for a brand-new Google account until `role` is supplied |
| `/api/v1/auth/refresh` | POST | Public | `{ refreshToken }` |
| `/api/v1/auth/forgot-password` | POST | Public | `{ email }` — mails a 6-digit OTP |
| `/api/v1/auth/verify-reset-otp` | POST | Public | `{ email, code }` → `{ resetToken }` |
| `/api/v1/auth/reset-password` | POST | Public | `{ token, newPassword }` — token comes from verify-reset-otp, never from the email |
| `/api/v1/auth/verify-email` | POST | Public | `{ token }` |
| `/api/v1/auth/resend-verification` | POST | Bearer JWT | — |
| `/api/v1/auth/me` | GET | Bearer JWT | — |

`AuthResponse`: `{ accessToken, refreshToken, user: { id, email, name, role, avatarUrl?, emailVerified, createdAt } }`.

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
- `notes/folder-structure.md` — full target structure for all phases

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
