# CLAUDE.md — Backend (Kotlin + Spring Boot)

**Status:** Not started. This folder is empty; scaffolding pending.

## Role in the project

The Spring Boot backend is the **single public API** for the website and mobile app. It owns users, events, participants, photo uploads, payments, and the marketplace. It delegates all ML work to `ai-api` — clients never call `ai-api` directly.

See root `CLAUDE.md` for monorepo-wide rules and `docs/project-vision.md` for product context.

## Stack

| Concern | Choice |
|---------|--------|
| Language | Kotlin (JVM 17+) |
| Framework | Spring Boot 3.x |
| Build | Gradle (Kotlin DSL — `build.gradle.kts`) |
| Auth | JWT bearer tokens |
| DB | PostgreSQL 16 (shared RDS instance with `ai-api`; separate schema) |
| Storage | AWS S3 |
| Async | Spring `@Async` + WebSocket for real-time upload notifications |
| Hosting | AWS EC2 |

## Module boundaries

- **Website + mobile → this backend only.** Never bypass to `ai-api`.
- **This backend → ai-api via server-to-server with API key** (`sk_backend_...`, scopes `*`, rate tier `internal`).
- **Event isolation:** every `ai-api` face call MUST include `event_id`.
- **Confidence thresholds:** apply per-event here, not in `ai-api`.

## What to read before coding

| Document | Why |
|----------|-----|
| `../docs/project-vision.md` | Product vision, feature matrix, user journeys |
| `../docs/IMPLEMENTATION_PLAN.md` | Phased roadmap — section 2 covers this backend |
| `../ai-api/docs/integration-architecture.md` | Boundary between backend and ai-api |
| `../ai-api/docs/integration-contracts.md` | Exact request/response shapes for ai-api calls |
| `../docs/api-keys.md` | Backend API key (gitignored) |

## First-milestone files (when scaffolding starts)

```
backend/
├── src/main/kotlin/com/quickpitik/
│   ├── QuickPitikApplication.kt
│   ├── config/          (SecurityConfig, CorsConfig, AiApiConfig, S3Config)
│   ├── controller/      (Auth, User, Event, Participant, Photo, Search, Order)
│   ├── service/         (per-domain services; AiApiClient wraps ai-api calls)
│   ├── repository/      (Spring Data JPA)
│   ├── model/           (entities — Kotlin data classes for DTOs, JPA entities for persistence)
│   └── dto/
└── src/test/kotlin/...
```

> Use the `kotlin-spring` and `kotlin-jpa` Gradle plugins so Spring/JPA can subclass `final` Kotlin classes without forcing `open` everywhere.

Flesh this file out as the backend comes online — replace stubs with real conventions, build commands, and gotchas.

## Working notes live in Obsidian

See `QuickPitik Vault/backend/` for feature designs, DB schema sketches, Spring learning notes, and todos.
