# AGENTS.md — Desktop App (Electron)

**Status:** Already built. Lives in a separate working directory.

## Actual location

```
Github Repo: https://github.com/theo2815/Batch-My-Photos.git
```

The local clone path differs per team member — resolve it from `../docs/team-workspaces.md` (match `git config user.email`, then confirm the directory exists).

**Access rule:** only when the user explicitly asks. Do NOT pre-emptively read or edit files in that path. Treat it as a separate environment. This `AGENTS.md` exists only to document the integration contract from this monorepo's point of view.

## Role in the project

A professional desktop tool for photographers handling large volumes (5,000–15,000 images per event). Reduces post-event sort from 1–2 hours to 5–10 seconds.

Features:
- Photographer login + profile.
- Automatic blur detection via `ai-api`.
- Automatic batch sorting into upload-ready folders.
- Batch processing without manual review.
- Upload a whole shoot folder to a QuickPitik event (signs in to the Spring Boot backend as a photographer through a browser hand-off; feature-flagged).

Scope: desktop does blur, sort, and upload to QuickPitik. No face search, no bib recognition, no marketplace.

## Module boundaries

- **Desktop is the ONE client that calls `ai-api` directly.** Uses its own restricted API key — scopes `blur:read` + `jobs:read`. The exact key, scopes, and rate tier are canonical in `../docs/api-keys.md`; don't restate them elsewhere.
- Desktop has its **own backend and database**, separate from the Spring Boot web/mobile backend.
- **Desktop → Spring Boot backend as a signed-in photographer, for the upload workflow only.** No API key and no ai-api in that path: a one-time-code sign-in (`POST /auth/desktop/exchange`), then the same photographer upload endpoints the website uses. Contract: `../docs/desktop-upload-integration-guide.md`.

## What to read before coding

| Document | Why |
|----------|-----|
| `../docs/project-vision.md` | Desktop journey (Journey 2: Photographer After Event) |
| `../docs/desktop-blur-detection-integration-guide.md` | Electron → ai-api specifics |
| `../docs/api-keys.md` | Desktop API key (gitignored) |
| `../ai-api/docs/integration-contracts.md` | Request/response shapes for blur endpoints |
| `../docs/desktop-upload-integration-guide.md` | Sign-in hand-off + upload contract for the Upload view |

## Working notes live in Obsidian

See `QuickPitik Vault/desktop/` for feature logs, ai-api integration notes, and todos.
