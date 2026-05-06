# CLAUDE.md — QuickPitik (Root)

Global rulebook for AI agents and contributors working in the `capstone-project` monorepo. This file is the **single source of truth for project-wide rules**. Module-specific details live in per-module `CLAUDE.md` files and the repository docs.

---

## Project Vision

**QuickPitik** is a marathon photography ecosystem for Cebu, Philippines. It combines a **Camera → Mobile → Cloud → Marketplace** pipeline with AI-powered quality filtering and search to solve two problems simultaneously:

- **Photographers** can't upload in real time, manually sort thousands of photos (1–2 hrs), and have no blur-culling tool.
- **Runners** can't find their own photos in a sea of thousands and have no local platform to buy them.

QuickPitik delivers: real-time camera tethering, desktop-only AI blur culling (BatchMyPhotos), face recognition + bib-number search for runners on web and mobile, and a marketplace. Four products serve these flows: a mobile app, a website, an already-built Electron desktop app, and a shared AI inference service (`ai-api`). Full details live in `docs/project-vision.md`.

**North-star outcomes**
- Photographers: post-event sort time from 1–2 hours → 5–10 seconds.
- Runners: photos appear within minutes and are searchable by selfie or bib number.

---

## Monorepo Layout

| Product | Stack | Status | Module CLAUDE.md |
|---------|-------|--------|------------------|
| `ai-api/` | FastAPI + Celery (Python 3.11+) | Phases 1–6 complete; hardening in progress | `ai-api/CLAUDE.md` |
| `backend/` | Spring Boot (Kotlin) | Not started | `backend/CLAUDE.md` (stub) |
| `website/` | Next.js on Vercel | UI scaffolding in progress | `website/CLAUDE.md` (stub) |
| `mobile/` | Kotlin (Android first) | Not started | `mobile/CLAUDE.md` (stub) |
| `desktop/` | Electron | Already built (lives at `C:\Users\Theo Cedric Chan\Documents\Start Up project\BatchMyPhotos`) | `desktop/CLAUDE.md` (stub) |

Repository-level docs are in `docs/`:
- `docs/project-vision.md` — authoritative vision, user journeys, feature matrix
- `docs/IMPLEMENTATION_PLAN.md` — phased roadmap across all modules
- `docs/api-keys.md` — **sensitive, gitignored** — API keys for desktop + backend
- `docs/desktop-blur-detection-integration-guide.md` — Electron → ai-api integration

---

## Module Boundaries (who talks to whom)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Mobile App  │  │   Website    │  │  Desktop App │
│  (Kotlin)    │  │  (Next.js)   │  │  (Electron)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌────────────────────────────┐  ┌──────────────────┐
│  Spring Boot Backend       │  │  Desktop Backend │
│  (users, events, payments) │  │  (own DB)        │
└────────────┬───────────────┘  └────────┬─────────┘
             │                           │
             ▼                           ▼
┌──────────────────────────────────────────────────┐
│  ai-api  (blur / face / bib / batch)             │
└──────────────────────────────────────────────────┘
```

**Hard rules:**

1. **Mobile and website NEVER call `ai-api` directly.** They call the Spring Boot backend; the backend holds the `ai-api` key and proxies inference.
2. **Desktop app is the one exception** — it calls `ai-api` directly with its own restricted API key (scopes: `blur:read`, `jobs:read`).
3. **Each backend owns its own domain data.** `ai-api` is stateless about events/users/participants — it only stores face embeddings tagged with `api_key_id` + `event_id`.
4. **Confidence thresholds are a backend concern.** `ai-api` returns raw scores; each backend applies its own per-event threshold.
5. **Event isolation must always be enforced.** Any `faces/enroll` or `faces/search` call MUST pass `event_id` — this is how cross-event data leakage is prevented.

Full integration contracts: `ai-api/docs/integration-architecture.md`, `ai-api/docs/integration-contracts.md`.

---

## Global Rules

These apply across every module. Module-specific rules (Python layering, Spring package conventions, React file structure) live in each module's own `CLAUDE.md`.

### Documentation
- Canonical, versioned docs live in `docs/` (repo-wide) and `<module>/docs/` (module-specific). Keep terse and accurate.
- Exploratory thinking, ADRs-in-progress, daily logs, and learning notes live in the Obsidian vault (see below).
- When a vault note stabilizes into canonical knowledge, **promote it** into the appropriate `docs/` location and leave a stub in the vault linking to the repo path.
- Never add `*.md` files to the repo unless explicitly requested or clearly required.

### Secrets & Configuration
- `.env` files, API keys, `WEBHOOK_SECRET_KEY`, and `docs/api-keys.md` are **gitignored**. Never commit them.
- Never hardcode URLs, thresholds, or secrets — read from env / module config.
- Treat the values in `docs/api-keys.md` as production-like credentials even in dev.

### Git & Commits
- Create NEW commits; never amend published commits without explicit user instruction.
- Never use `--no-verify` or bypass signing unless the user asks.
- Never force-push `main`.
- Stage specific files; avoid `git add -A`.

### Cross-module changes
- Changes that span `ai-api` ↔ backend ↔ client require explicit coordination — update the integration docs in the same commit.
- Breaking API changes go under a new version prefix (`/api/v2/...`); existing clients must not be silently broken.

### AI agent behavior
- Always consult the Obsidian vault rules below before starting non-trivial work.
- Prefer editing existing files over creating new ones.
- Task tracking: use the task system for multi-step work; promote completed decisions/logs into the vault.

---

## Architectural Principles

1. **`ai-api` is an internal service.** Never expose it to end-user apps directly.
2. **One API key per backend, per environment.** `api_key_id` is the tenant boundary in `ai-api`.
3. **Scopes are least-privilege.** Desktop gets `blur:read` + `jobs:read` only. Spring Boot gets the full scope set.
4. **Async batch = blob-store, not base64.** Image bytes never go on the Celery queue; workers read from `BLOB_STORE_PATH`.
5. **Webhooks are HMAC-signed.** Consumers must verify `X-QuickPitik-Signature`.
6. **Photos are private by default.** Runners only see their own matched photos; public galleries are opt-in per event.
7. **Scaling path is known, not premature.** Start with pgvector on RDS + direct S3. Migrate to Qdrant or add CloudFront only when metrics demand it (see `docs/project-vision.md#scaling-path`).

---

## Custom Skills

Project-specific skills live in the Obsidian vault under `Claude Skills/`. Agents MUST consult these skills when their trigger conditions match — they encode the user's standards and override default behavior.

| Skill | Path | When to apply |
|-------|------|---------------|
| **Frontend Design** | `C:\Users\Theo Cedric Chan\Documents\Obsidian Vault\QuickPitik Vault\Claude Skills\Frontend Design.md` | Any task that creates, redesigns, or polishes a UI/UX — components, pages, layouts, styling, animations, design systems. Applies across `website/`, `mobile/`, and `desktop/` modules. |
| **Document Skill** | **Skill (rulebook):** `C:\Users\Theo Cedric Chan\Documents\Obsidian Vault\QuickPitik Vault\Claude Skills\Document Skill.md`<br>**Workspace (drafts, facts, logs):** `C:\Users\Theo Cedric Chan\Documents\Obsidian Vault\QuickPitik Vault\Documentation for capstone paper\` | Any task that writes, drafts, reviews, or asks about the **capstone paper** — currently **SRS** (active this semester), **SDD** (next semester). Triggers on phrases like "the SRS", "the SDD", "draft §X.Y", "my paper", "review my section", "promote to Papers-For-Capstone", "what's left on the SRS". **Mandatory protocol:** read the skill file in full, then run the pre-draft ritual it specifies (workspace `README.md` → relevant `facts/*.md` → `open-questions.md` → `adviser-log.md` → draft file). The skill enforces IEEE-830 spec voice, hard rules against fact invention (`[NEEDS USER INPUT — Q-NNN]` placeholder + pause), template fidelity (cover → §3.4 only, numbering gaps preserved), the FORBIDDEN PATTERNS list, and a 10-point self-audit checklist before any section is declared "Draft done." Final assembled SRS is promoted to `Papers-For-Capstone\SRS-QuickPitik.md` for external markdown → Word → PDF conversion. |

### How to use a skill

1. **Detect the trigger.** When the user's request matches a skill's trigger phrases or domain (UI/UX work → Frontend Design; capstone paper writing → Document Skill), the skill applies.
2. **Read the skill file in full** before producing any work it covers. Do not rely on summaries or memory — skills encode specific principles, prohibitions, and rituals that override defaults.
3. **Run any pre-task ritual the skill specifies.** The Document Skill, for example, mandates reading the workspace `README.md`, the relevant `facts/*.md`, `open-questions.md`, and `adviser-log.md` before drafting any SRS section. Skipping the ritual is a protocol failure.
4. **Apply the skill's guidance throughout the task.** Skills override default approaches — that is the point. Frontend Design overrides "use Tailwind / shadcn defaults"; Document Skill overrides "draft from agent assumptions."
5. **Honor the skill over defaults.** If a default approach conflicts with the skill, the skill wins.

If a skill applies to the user's request and is not consulted, the work is incomplete. Re-read the skill at the start of each task it covers — do not assume prior context carries over.

---

## External Working Directories

Some code referenced by this project lives outside the monorepo. Treat each as a separate environment.

| Path | Project | Access rule |
|------|---------|-------------|
| `C:\Users\Theo Cedric Chan\Documents\Start Up project\BatchMyPhotos` | Desktop app (Electron) | **Only access when explicitly requested by the user.** Do NOT assume or automatically modify files. Treat as a separate environment outside the main repo. |

If the user asks you to work in BatchMyPhotos, change directory and work there as normal. Do not pre-emptively read or edit files in that path based on monorepo activity.

---

## Obsidian Vault — The Second Brain

The Obsidian vault is the **primary external knowledge system** for this project — session memory, tasks, decisions, and module-specific working notes that don't belong in the repo.

**Vault path:**

```
C:\Users\Theo Cedric Chan\Documents\Obsidian Vault\QuickPitik Vault
```

**The vault has its own `CLAUDE.md` — read it at the start of every non-trivial session.** It owns the second-brain ritual, sync rules (`tasks.md`, `decisions.md`, `index.md`, `VAULT-INDEX.md`, `notes/`), naming conventions, and folder layout. This repo CLAUDE.md does not duplicate that — go to the source.

### Session start ritual (MANDATORY)

At the start of every session, before doing any other work, the agent MUST:

1. **Read the vault first.** Open the vault `CLAUDE.md` and `VAULT-INDEX.md` to load the current second-brain state (module status, open tasks, recent decisions).
2. **Ask the user which module(s) to work on.** Present the choices and let the user pick **one or more**:
   - `ai-api`
   - `backend`
   - `website`
   - `mobile`
   - `desktop`
3. **After the user selects**, read that module's `tasks.md` (and `index.md` / `decisions.md` if relevant) from the vault and **show the tasks** for the chosen module(s). If the user picked multiple, group the tasks per module.
4. Only after the user confirms which task(s) to tackle should the agent begin implementation work.

Do not skip this ritual. Do not start editing code, planning, or searching the repo before the vault has been read and the user has chosen a module + task.

**Quick orientation (canonical source: vault's `CLAUDE.md` + `VAULT-INDEX.md`):**

```
QuickPitik Vault/
├── CLAUDE.md            (vault rules + session ritual — READ FIRST)
├── VAULT-INDEX.md       (status dashboard + module map)
├── _templates/          (decision-log, daily-note, feature-doc)
├── _project/            (cross-cutting: vision/, architecture/, decisions.md)
├── ai-api/   { index, tasks, decisions, notes/ }
├── backend/  { index, tasks, decisions, notes/ }
├── website/  { index, tasks, decisions }
├── mobile/   { index, tasks, decisions }
└── desktop/  { index, tasks, decisions }
```

**Vault vs. repo docs:**
- **Repo** (`docs/`, `<module>/docs/`): canonical, versioned, audience-facing. Terse and accurate.
- **Vault**: working memory, exploration, decisions-in-progress, learning notes. Private.
- When a vault note stabilizes, **promote** it to the repo and leave a stub in the vault pointing to the repo path.

---

## Where to Find Specifics

| I want to know... | Read |
|---|---|
| The product vision and user journeys | `docs/project-vision.md` |
| Phased implementation roadmap | `docs/IMPLEMENTATION_PLAN.md` |
| ai-api architecture, endpoints, conventions | `ai-api/CLAUDE.md` and `ai-api/docs/` |
| How backends integrate with ai-api | `ai-api/docs/integration-architecture.md`, `ai-api/docs/integration-contracts.md` |
| API keys and how each client authenticates | `docs/api-keys.md` (gitignored) |
| Desktop → ai-api specifics | `docs/desktop-blur-detection-integration-guide.md` |
| Working notes, decisions, module status | Obsidian vault — start at `CLAUDE.md` and `VAULT-INDEX.md` |

---

## User & Environment Context

- **User:** theocedric.chan@cit.edu — CIT-U capstone student, Cebu, Philippines.
- **Today's date:** see `currentDate` in the conversation context; convert relative dates to absolute before saving to memory or vault.
- **Platform:** Windows 11, bash shell available, PowerShell available.
- **Target market:** Marathon and running-event photography in Cebu (Philippines-first, regional expansion later).

---

## Goal

This `CLAUDE.md` is the **central control system** for the project. The Obsidian vault is the **second brain** for continuity, decisions, and knowledge. Together they let any agent:

- Resume work seamlessly across sessions.
- Follow consistent architecture and module boundaries.
- Stay aligned with the project vision without re-deriving it each time.

If something here conflicts with a module-specific `CLAUDE.md`, the module's rules win for that module. If something conflicts with `docs/project-vision.md`, update one of them — they must agree.
