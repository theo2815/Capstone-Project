# CLAUDE.md — Mobile App (Kotlin)

**Status:** Skeleton Initialized (Active Development). Design Tokens configured; MVVM package stubs mapped.

## Build and Test Commands
* Build app: `.\gradlew.bat assembleDebug` (from the `mobile/` directory)
* Run unit tests: `.\gradlew.bat test`
* Clean build cache: `.\gradlew.bat clean`

## Role in the project

The mobile app serves two audiences:

- **Photographers:** camera tethering (WiFi/USB), real-time upload during events, upload progress UI.
- **Runners:** event browsing, face-selfie search, bib-number search, push notifications, watermarked previews, purchase + download.

See root `CLAUDE.md` for monorepo-wide rules and `docs/project-vision.md` for user journeys.

## Stack

| Concern | Choice |
|---------|--------|
| Language | Kotlin |

| Platform | Android first, iOS planned |
| Distribution | Google Play Store |
| Auth | JWT from backend |

## Module boundaries

- **Mobile → Spring Boot backend ONLY.** Never call `ai-api` directly.
- Camera tethering happens on-device; upload is direct-to-S3 with a backend-signed URL.

## What to read before coding

| Document | Why |
|----------|-----|
| `../docs/project-vision.md` | Photographer and runner journeys |
| `../docs/IMPLEMENTATION_PLAN.md` | Phased roadmap — section 3 covers mobile |

## Working notes live in Obsidian

See `QuickPitik Vault/mobile/` for feature designs, architecture choices (MVVM? Clean Architecture?), Kotlin learning notes, and todos.
