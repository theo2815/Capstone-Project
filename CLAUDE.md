# CLAUDE.md — QuickPitik (Root)

Global rulebook for AI agents and contributors in the `capstone-project` monorepo. **Project-wide rules only.** Volatile status, surface-specific detail, and rationale live elsewhere — see the routing table at the bottom. Keep this file under ~200 lines: past that, adherence drops.

**One home per fact:** a fact that changes when the code changes lives with the code (or as a pointer to the command/config that generates it); *why* we chose something lives in the vault. Never copy a value into a second doc — point to its home.

---

## Project Vision

**QuickPitik** — a marathon photography ecosystem for Cebu, Philippines. **Camera → Mobile → Cloud → Marketplace**, with AI blur culling on the desktop and face/bib search on web + mobile. It solves two problems: photographers can't upload in real time or cull blur; runners can't find their own photos in a sea of thousands or buy them locally.

Four products + one service: a mobile app, a website, an Electron desktop app (BatchMyPhotos, external repo), and a shared `ai-api` inference service. Full vision, user journeys, and feature matrix: `docs/project-vision.md`.

**North-star:** photographer post-event sort 1–2 hrs → 5–10 s; runner photos searchable by selfie or bib within minutes.

---

## Monorepo Layout

| Product | Path | Stack | Module rulebook |
|---------|------|-------|-----------------|
| ai-api | `ai-api/` | FastAPI + Celery (Python 3.11+) | `ai-api/CLAUDE.md` |
| backend | `backend/` | Spring Boot (Kotlin, JDK 21) | `backend/CLAUDE.md` |
| website | `website/` | Next.js on Vercel | `website/CLAUDE.md` |
| mobile | `mobile/` | Kotlin (Android first) | `mobile/CLAUDE.md` |
| desktop | external repo | Electron (BatchMyPhotos) | `desktop/CLAUDE.md` (stub) |

**Status, test counts, and ship-state are deliberately NOT recorded here** — they go stale the moment code changes. For the live dashboard read the vault `VAULT-INDEX.md` / `ROLE-STATUS.md`; for counts run the module's test command (`./gradlew test` for backend, `pytest` for ai-api, etc.).

---

## Module Boundaries (who talks to whom)

**Data flow:** Mobile / Website → Spring Boot backend → `ai-api` (face / bib). Desktop → `ai-api` directly (blur / batch). The detailed enforcement contract loads on demand via `.claude/rules/ai-api-boundary.md` when you touch `backend/`, `website/`, or `mobile/`.

**Hard rules (always in force):**

1. **Mobile and website NEVER call `ai-api` directly.** They call the Spring Boot backend; the backend holds the `ai-api` key and proxies inference. (The backend may also serve face/bib via AWS Rekognition — selected by `AI_PROVIDER` — see the boundary rule.)
2. **Desktop is the one exception** — it calls `ai-api` directly with its own restricted API key (scopes: `blur:read`, `jobs:read`).
3. **Each backend owns its own domain data.** `ai-api` is stateless about events/users — it stores only face embeddings tagged with `api_key_id` + `event_id`.
4. **Confidence thresholds are a backend concern.** `ai-api` returns raw scores; each backend applies its own per-event threshold.
5. **Event isolation must always be enforced.** Any `faces/enroll` or `faces/search` call MUST pass a non-null `event_id` — this prevents cross-event leakage.
6. **Blur detection is desktop-only.** Web + mobile upload paths MUST NOT call any blur endpoint (no `/blur/detect`, no `BLUR_REJECTED` gate, no `blurScore` writes). The backend's `AiApiClient` intentionally has no `blurDetect()` — do not re-add it. Rationale: vault `backend/decisions.md` (2026-05-18), `website/decisions.md` (2026-05-06).
7. **`AI_API_ENABLED=false` is the dev default** (env `AI_API_ENABLED`, `application.yml`). When off: photo/selfie upload still works (no embeddings/OCR, `aiDetectionStatus="none"`), face-search short-circuits to 503. It is the single point of control — don't paper over it with mocks or new flags.

Full contracts: `ai-api/docs/integration-architecture.md`, `ai-api/docs/integration-contracts.md`.

---

## Global Rules

Module-specific conventions (Python layering, Spring packages, React structure) live in each module's own `CLAUDE.md`.

**Documentation**
- Canonical, versioned docs live in `docs/` (repo-wide) and `<module>/docs/`. Keep terse and accurate.
- Exploration, ADRs-in-progress, daily logs, and decisions live in the Obsidian vault.
- When a vault note stabilizes into canonical knowledge, **promote it** to `docs/` and leave a vault stub pointing at the repo path — never keep two live copies.
- Never add `*.md` files to the repo unless explicitly requested or clearly required.

**Secrets & Configuration**
- `.env` files, API keys, `WEBHOOK_SECRET_KEY`, and `docs/api-keys.md` are **gitignored**. Never commit them.
- Never hardcode URLs, thresholds, or secrets — read from env / module config.
- `docs/api-keys.md` is the **canonical source** for keys, scopes, and rate tiers; treat its values as production-like even in dev.

**Git & Commits**
- Create NEW commits; never amend published commits without explicit instruction.
- Never use `--no-verify` or bypass signing unless the user asks. Never force-push `main`.
- Stage specific files; avoid `git add -A`.

**Cross-module changes**
- Changes spanning `ai-api` ↔ backend ↔ client require coordination — update the integration docs in the same commit.
- Breaking API changes go under a new version prefix (`/api/v2/...`); existing clients must not be silently broken.

**AI agent behavior**
- Prefer editing existing files over creating new ones.
- Use the task system for multi-step work; promote completed decisions/logs into the vault.

---

## Engineering Discipline — Apply Before Coding (MANDATORY)

Govern **how** any agent writes code here, and must be satisfied **before the first line** — not as after-the-fact review. Bias toward caution over speed; for genuinely trivial tasks (typo, single-line edit, lookup) use judgment. If these conflict with a module `CLAUDE.md`, follow the stricter rule.

1. **Confirm alignment before acting.** If a prompt is unclear or you're not fully confident you understood it, ask first — every time, never guess. Restate your understanding in a sentence; surface the exact ambiguity and the interpretations you're choosing between. Skip only for genuinely trivial, unambiguous requests. Better one extra question than the wrong build.
2. **Think before coding.** State assumptions explicitly. If multiple interpretations exist, present them — don't pick silently. If a simpler approach exists, say so and push back. If something's unclear, stop and name it.
3. **Simplicity first.** Minimum code that solves the problem, nothing speculative — no features beyond what was asked, no abstractions for single-use code, no unrequested flexibility, no error handling for impossible cases. If 200 lines could be 50, rewrite. Ask: "would a senior engineer call this overcomplicated?"
4. **Surgical changes.** Touch only what you must. Don't improve, refactor, or reformat adjacent code that isn't broken; match existing style. Remove only the orphans YOUR change creates; flag pre-existing dead code, don't delete it. Every changed line should trace directly to the request.
5. **Goal-driven execution.** Turn tasks into verifiable goals ("fix the bug" → "write a test that reproduces it, then make it pass"). State a brief plan with a verify check per step, and loop until verified. Weak criteria ("make it work") force constant clarification; strong ones let you work independently.

**Working if:** fewer unnecessary diff lines, fewer overcomplication rewrites, clarifying questions before implementation rather than after mistakes.

---

## Architectural Principles

1. **`ai-api` is an internal service.** Never expose it to end-user apps directly.
2. **One API key per backend, per environment.** `api_key_id` is the tenant boundary in `ai-api`.
3. **Scopes are least-privilege.** Desktop gets `blur:read` + `jobs:read` only; Spring Boot gets the full scope set. (Exact values: `docs/api-keys.md`.)
4. **Async batch = blob-store, not base64.** Image bytes never go on the Celery queue; workers read from `BLOB_STORE_PATH`.
5. **Webhooks are HMAC-signed.** Consumers must verify `X-QuickPitik-Signature`.
6. **Photos are private by default.** Runners only see their own matched photos; public galleries are opt-in per event.
7. **Scaling path is known, not premature.** Start with pgvector on RDS + direct S3; migrate to a dedicated vector DB or add a CDN only when metrics demand it.

---

## Custom Skills

Project skills live in the vault under `Claude Skills/`. **Read the matching skill in full before producing work it covers — skills encode the user's standards and override defaults (Tailwind/shadcn defaults, "draft from assumptions", etc.).**

- **UI work on `website/` or `desktop/`** → the Frontend Design skill loads automatically via `.claude/rules/frontend-design.md`.
- **UI work on `mobile/`** → the Mobile Design skill loads via `.claude/rules/mobile-design.md`.
- **Capstone paper (SRS this semester, SDD next)** — phrase-triggered ("the SRS", "draft §X.Y", "review my section", "my paper"): read `…\QuickPitik Vault\Claude Skills\Document Skill.md` in full, then run its pre-draft ritual (workspace `README.md` → `facts/*.md` → `open-questions.md` → `adviser-log.md` → draft). Final SRS is promoted to `Papers-For-Capstone\SRS-QuickPitik.md`.

---

## External Working Directories

| Path | Project | Access rule |
|------|---------|-------------|
| `…\Start Up project\BatchMyPhotos` | Desktop app (Electron) | **Only access when explicitly requested.** Separate environment — do not pre-emptively read or edit based on monorepo activity. |

BatchMyPhotos is itself a multi-project repo (`desktop/` Electron · `backend/` its own Express server · `website/`) — **NOT** the QuickPitik Spring Boot `backend/` or Next.js `website/`. When the user picks `desktop`, ask which sub-project, then **lock the session into `…\BatchMyPhotos\`**. The only exception is blur-detection work, which spans BatchMyPhotos ↔ `ai-api` here.

---

## Obsidian Vault — The Second Brain

Session memory, tasks, decisions, and working notes that don't belong in the repo.
**Path:** `C:\Users\Theo Cedric Chan\Documents\Obsidian Vault\QuickPitik Vault`

**The vault owns its own `CLAUDE.md`** — the second-brain ritual, sync rules, naming conventions, and folder layout. Read it; this file does not duplicate it.

**Session start (MANDATORY):**
1. Read the vault `CLAUDE.md` + `VAULT-INDEX.md` to load current state (status, open tasks, recent decisions).
2. Ask which module(s) to work on: `ai-api` · `backend` · `website` · `mobile` · `desktop`.
3. Read that module's `tasks.md` from the vault and show its tasks. Begin implementation only after the user confirms a task.

Do not start editing, planning, or searching before the vault is read and a module + task chosen.

**Vault vs repo:** vault = working memory, rationale, and the status dashboard (private); repo `docs/` = canonical, versioned, audience-facing. When a vault note stabilizes, promote it to `docs/` and leave a stub — never two live copies.

---

## Where to Find Specifics (go read — not preloaded)

| I want to know… | Read |
|---|---|
| Current status, test counts, ship-state | vault `VAULT-INDEX.md` / `ROLE-STATUS.md`; or run the module's test command |
| Product vision + user journeys | `docs/project-vision.md` |
| Phased roadmap (frozen history + live pointers) | `docs/IMPLEMENTATION_PLAN.md` |
| ai-api architecture, endpoints, conventions | `ai-api/CLAUDE.md` and `ai-api/docs/` |
| How backends integrate with ai-api | `ai-api/docs/integration-architecture.md`, `ai-api/docs/integration-contracts.md` |
| API keys, scopes, rate tiers | `docs/api-keys.md` (gitignored — canonical) |
| Desktop → ai-api blur specifics | `docs/desktop-blur-detection-integration-guide.md` |
| Running ai-api locally, end-to-end | `docs/run-ai-api-locally.md` |
| Why a decision was made (ADRs) | vault `_project/decisions.md` and module `decisions.md` |

---

## User & Environment

- **User:** theocedric.chan@cit.edu — CIT-U capstone student, Cebu, Philippines.
- **Today's date:** see `currentDate` in the conversation context; convert relative dates to absolute before saving to memory or the vault.
- **Platform:** Windows 11; PowerShell and bash available.
- **Target market:** marathon and running-event photography in Cebu (Philippines-first, regional expansion later).

If this file conflicts with a module `CLAUDE.md`, the module's rules win for that module. If it conflicts with `docs/project-vision.md`, fix one — they must agree.
