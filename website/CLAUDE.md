# CLAUDE.md — Website (Next.js)

**Status:** UI scaffolding in progress (`src/app/`, `src/components/`, `src/hooks/`, `src/lib/`, `src/store/`, `src/types/` exist). No backend wiring yet.

## Role in the project

The website is the non-app path for runners and the primary tool for event organizers. It provides:

- Runner search (face selfie, bib number) and the photo marketplace.
- Photographer dashboard, event management, sales analytics.
- Public event galleries.

See root `CLAUDE.md` for monorepo-wide rules and `docs/project-vision.md` for the feature matrix.

## Stack

| Concern | Choice |
|---------|--------|
| Framework | Next.js (App Router) |
| Language | TypeScript |
| Hosting | Vercel |
| Styling | Tailwind (see `src/app/globals.css`) |
| State | Store under `src/store/` |
| Auth | JWT from backend, stored client-side |

## Module boundaries

- **Website → Spring Boot backend ONLY.** Never call `ai-api` directly — the backend proxies all ML inference.
- Photos are served from S3 (via backend-signed URLs). Full-res downloads only after purchase verification.

## What to read before coding

| Document | Why |
|----------|-----|
| `../docs/project-vision.md` | Runner + photographer journeys, feature matrix |
| `../docs/IMPLEMENTATION_PLAN.md` | Phased roadmap — section 4 covers this website |

## Current scaffold

```
website/src/
├── app/            (pages)
├── components/     (layout/, ui pieces)
├── hooks/
├── lib/
├── store/
└── types/
```

## Working notes live in Obsidian

See `QuickPitik Vault/website/` for page designs, component patterns, design-system decisions, and todos.
