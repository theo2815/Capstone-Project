# CLAUDE.md — Backend (Kotlin + Spring Boot)

**Status:** Phase A — auth scaffold complete (2026-05-05). Broader build deferred until website frontend reaches full lock state. See vault `backend/decisions.md`.

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
.\gradlew.bat test
```

Unit tests run without Postgres. End-to-end is verified via `curl` (see Smoke Test below).

### Stop Postgres

```powershell
docker compose down            # keeps data volume
docker compose down -v         # wipes data (forces V1 to re-run on next boot)
```

## Environment variables

All env vars have dev-friendly defaults in `application.yml` so the app boots out of the box. Override in production via OS env, `.env`, or a `application-prod.yml` (gitignored).

| Variable | Default | Production behavior |
|----------|---------|---------------------|
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
| `AI_API_ENABLED` | `false` | Set `true` when ai-api is wired and you want face/bib indexing on upload + selfie quality gate + runner face-search. When `false`, every server-side ai-api call is skipped — photographer upload + selfie upload still work, face-search returns 503. See vault `backend/decisions.md` 2026-05-18 master-switch ADR. |

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

# 5. Forgot password — check console for [EMAIL STUB] log line
curl -X POST http://localhost:8080/api/v1/auth/forgot-password \
  -H "Content-Type: application/json" \
  -d '{"email":"runner@test.local"}'

# 6. Reset with the logged token
curl -X POST http://localhost:8080/api/v1/auth/reset-password \
  -H "Content-Type: application/json" \
  -d '{"token":"<RESET_TOKEN_FROM_LOG>","newPassword":"newpassword123"}'

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
| `/api/v1/auth/refresh` | POST | Public | `{ refreshToken }` |
| `/api/v1/auth/forgot-password` | POST | Public | `{ email }` |
| `/api/v1/auth/reset-password` | POST | Public | `{ token, newPassword }` |
| `/api/v1/auth/me` | GET | Bearer JWT | — |

`AuthResponse`: `{ accessToken, refreshToken, user: { id, email, name, role, avatarUrl?, createdAt } }`.

Roles are UPPERCASE in JSON: `"ADMIN" | "PHOTOGRAPHER" | "RUNNER"`. `ADMIN` is not creatable via `/auth/register` — only via `BootstrapAdminRunner` on first boot, or admin-promotion endpoints (Phase G).

## Implementation notes

- **Refresh tokens** are opaque random 32-byte base64url strings, hashed with SHA-256 before persistence. **Rotated on every refresh** (parent token revoked, new one issued). On `confirmReset`, all of a user's refresh tokens are revoked to log out other sessions.
- **Reset tokens** same generation, 15-min expiry, one-shot use.
- **Access tokens** are JWTs (HS256) carrying `sub=userId`, `email`, `role` claims. 15-min TTL.
- **Forgot-password** is intentionally silent if the email doesn't exist (anti-enumeration). The endpoint always returns the same generic message.
- **Email stub**: `EmailService` logs the reset link to stdout. Replace with Resend/SES/SendGrid in a future iteration (icebox).
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
