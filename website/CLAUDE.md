# CLAUDE.md — Website (Next.js)

**Status:** All four roles (Guest · Runner · Photographer · Admin) feature-complete; last reconciled 2026-08-16. Wired to the live Spring Boot backend throughout — **there is no mock fallback and no `NEXT_PUBLIC_BACKEND_LIVE` gate** (that flag was never wired and is dead code; see vault `website/decisions.md` 2026-05-18). Per-surface truth lives in vault `ROLE-STATUS.md`.

Baselines to hold: `npx tsc --noEmit` clean · `npm run lint` 0 errors / **300** warnings · `npm run build` succeeds. (Measured 2026-08-16 by stashing the working diff and re-running; the "~298" recorded here before that was already stale. If your count differs, re-measure the same way before assuming you added warnings.)

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

## UI Change Pre-flight (BLOCKING)

Before reporting any UI task complete — including bug fixes, new features, refactors, or token migrations — verify in a real browser:

1. **Mobile-first.** Open the route at 375 px and walk through the feature end-to-end. Quiet Studio is mobile-first; if it doesn't sing on a phone, it doesn't ship.
2. **Two more breakpoints.** Confirm `md` (≥768) and `lg` (≥1024) don't break.
3. **Sibling regression scan.** Open at least one *other* page in the same module that uses any component you touched. Touched `PhotoPreviewCard`? Open `/events/[slug]?browse=1`, the cart-modal flow, AND `/orders`. Touched `SiteHeader` or `AvatarDisc`? Spot-check every page that mounts them.
4. **Quiet Studio audit** (rules live in `QuickPitik Vault/website/notes/design-system.md`):
   - One `fresh` element per viewport (excluding tiny dots/indicators).
   - Display headlines normal case; mono kickers uppercase.
   - Every numeric carries `tnum`.
   - No legacy tokens (`charcoal`, `primary`, `teal`, `steel-blue`, `cool-gray`, `warm-white`) on redesigned pages.
5. **Modals/dialogs/lightboxes** render via `createPortal(content, document.body)`. See `notes/ui-pitfalls.md` for the containing-block trap that motivates this.
6. **State persistence.** Refresh the page once with the feature open — does Zustand-persisted state restore correctly? Does any in-flight URL state survive?

If you cannot actually open a browser in this environment, **say so explicitly**. Type-check + lint passing is necessary but not sufficient — do not claim success on UI work you have not visually verified.

## Before any UI change — read these

These three vault notes are the design contract for the website. Read the relevant ones before opening a file in `src/app/` or `src/components/`:

| Note | Why |
|------|-----|
| `QuickPitik Vault/website/notes/design-system.md` | Quiet Studio tokens, type system, hard rules. The single source of truth for "does this match the aesthetic." |
| `QuickPitik Vault/website/notes/components.md` | Locked + provisional component vocabulary. Reuse before recreating. |
| `QuickPitik Vault/website/notes/ui-pitfalls.md` | Append-only log of UI bugs that shipped, with root cause + how-to-avoid. Read before any non-trivial change. |

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
