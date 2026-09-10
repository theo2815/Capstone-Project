# AGENTS.md — Website (Next.js)

**Status:** All four roles (Guest · Runner · Photographer · Admin) feature-complete. Wired to the live Spring Boot backend throughout — **there is no mock fallback and no `NEXT_PUBLIC_BACKEND_LIVE` gate** (that flag was never wired and is dead code; see vault `website/decisions.md` 2026-05-18). Per-surface truth lives in vault `ROLE-STATUS.md`.

Baselines to hold: `npx tsc --noEmit` clean · `npm run lint` 0 errors, non-zero warnings baseline · `npm run build` succeeds. (The warning count drifts — the current baseline is in vault `website/tasks.md`; always re-measure on a clean tree before assuming you added warnings.)

## Role in the project

The website is the non-app path for runners and the primary tool for event organizers. It provides:

- Runner search (face selfie, bib number) and the photo marketplace.
- Photographer dashboard, event management, sales analytics.
- Public event galleries.

See root `AGENTS.md` for monorepo-wide rules and `docs/project-vision.md` for the feature matrix.

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

1. **Mobile-first.** Open the route at 375 px and walk through the feature end-to-end. Finish Line is mobile-first; if it doesn't sing on a phone, it doesn't ship.
2. **Two more breakpoints.** Confirm `md` (≥768) and `lg` (≥1024) don't break.
3. **Sibling regression scan.** Open at least one *other* page in the same module that uses any component you touched. Touched `PhotoPreviewCard`? Open `/events/[slug]?browse=1`, the cart-modal flow, AND `/orders`. Touched `SiteHeader` or `AvatarDisc`? Spot-check every page that mounts them.
4. **Finish Line audit** (rules live in `QuickPitik Vault/website/notes/design-system.md` — rewritten 2026-08-25):
   - One `fresh` element per viewport (excluding tiny dots/indicators).
   - Anton uppercase only via `.font-hero` (never in admin/dashboard — those headline in Archivo extrabold); mono kickers uppercase at the `<Kicker>`/`.kicker` floor.
   - Buttons come from `src/components/ui/button-styles.ts` (`BTN_PRIMARY/SECONDARY/DANGER/GHOST` + `BTN_SIZE`); text on green/ink/error fills is `text-surface`, never `text-bone`; disabled = `opacity-50 cursor-not-allowed`.
   - Every numeric carries `tnum`.
   - No legacy tokens (`charcoal`, `primary`, `teal`, `steel-blue`, `cool-gray`, `warm-white`) on redesigned pages.
5. **Modals/dialogs/lightboxes** render via `createPortal(content, document.body)`. See `notes/ui-pitfalls.md` for the containing-block trap that motivates this.
6. **State persistence.** Refresh the page once with the feature open — does Zustand-persisted state restore correctly? Does any in-flight URL state survive?

If you cannot actually open a browser in this environment, **say so explicitly**. Type-check + lint passing is necessary but not sufficient — do not claim success on UI work you have not visually verified.

## Before any UI change — read these

These three vault notes are the design contract for the website. Read the relevant ones before opening a file in `src/app/` or `src/components/`:

| Note | Why |
|------|-----|
| `QuickPitik Vault/website/notes/design-system.md` | The current design tokens, type system, and hard rules (rewritten 2026-08-25). The single source of truth for "does this match the aesthetic" — read it rather than assuming a token set. |
| `QuickPitik Vault/website/notes/components.md` | Locked + provisional component vocabulary. Reuse before recreating. |
| `QuickPitik Vault/website/notes/ui-pitfalls.md` | Append-only log of UI bugs that shipped, with root cause + how-to-avoid. Read before any non-trivial change. |

## What to read before coding

| Document | Why |
|----------|-----|
| `../docs/project-vision.md` | Runner + photographer journeys, feature matrix |
| `../docs/IMPLEMENTATION_PLAN.md` | Phase history + live roadmap (file created 2026-08-19 — this row had pointed at a nonexistent path) |

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

## Architecture & State Boundaries

- **Keep routes as composition boundaries.** `src/app/` owns Next.js pages, layouts, route parameters, and route-only components. Reusable components, hooks, stores, services, utilities, and types must not import from `app/`, including type-only imports. Put a shared type with its domain before reusing it across routes.
- **Use the existing domain folders.** Feature UI belongs in `components/<domain>/`; application shells belong in `components/layout/`; domain-independent primitives belong in `components/ui/`. Keep small helpers beside their only consumer. Extract a workflow when its responsibilities require independent maintenance or testing; do not create a service/repository/factory layer for every file or reorganize unrelated features during a fix.
- **Separate transport, orchestration, and calculations.** `lib/api.ts` owns shared HTTP/auth behavior; `lib/api-<domain>.ts` owns backend calls and response mapping. `hooks/use-<domain>*.ts` owns React query/mutation orchestration. Keep business calculations and validation as pure domain helpers in `lib/`, and cross-module domain contracts in `types/`. Generic utilities must have real cross-domain reuse. Shared UI primitives must not fetch domain data or read domain stores.
- **One canonical cache for server data.** Use TanStack Query for backend-owned records, with domain query keys scoped to the authenticated owner where appropriate. Mutations update or invalidate the relevant cache. Do not add another Zustand copy or custom fetch/cache layer for the same records. Existing override stores are transitional; do not extend that pattern to new features. Replace them only within an approved migration.
  Photographer and runner inboxes use `hooks/use-inbox-messages.ts`; bells and modals share owner/session-scoped infinite Query pages. WebSocket events invalidate those pages through `hooks/use-inbox-notifications-ws.ts`; do not push a second message list or synthesize read timestamps.
  Admin users, verification settings and per-user decision history use `hooks/use-admin-users.ts`; never mirror photographer Settings into an admin store or synthesize decision logs. Admin events use `hooks/use-admin-events.ts`; public catalogs use `hooks/use-public-events.ts` and never merge admin rows. Payout cycles use `hooks/use-admin-payouts.ts`; disputes use `hooks/use-admin-disputes.ts`; admin and photographer payout reports use `hooks/use-payout-reports.ts`. Their hooks own query keys, backend-confirmed cache updates, and invalidation; components own form state and success messages. Render server-issued activity and timestamps directly. The reusable-module ban on `app/` imports is enforced by ESLint.
- **Confirm consequential mutations with the backend.** Financial, admin, and destructive actions must await the API before committing status changes, decision logs, tombstones, or success messages. Return an explicit result or reject; callers must handle failure, retain retryable input, and disable duplicate submission while pending. Do not swallow failures or manufacture success. Optimistic updates for reversible actions require rollback and reconciliation.
- **Make client-state ownership explicit.** Local component state owns forms and transient workflow state; URL state owns shareable filters/selections; Zustand owns cross-component UI state, guest buffers, and explicitly resumable workflows. Before adding persisted state, document its owner (device, guest, or authenticated user), persistence key, expiry/version needs, and reset behavior. User-scoped state must reset through `resetUserScopedStores()` on auth transitions; only the explicit guest cart/saved-event buffer transfers into login.
- **Discard stale asynchronous results.** Before applying user-scoped responses or enabling background sync, capture the auth session before starting and verify its version, owner, and role still match before applying success, rollback, error, or chained-request effects. Use `lib/auth-session.ts` for imperative work; user ID alone cannot distinguish logout followed by login to the same account. Scope in-flight deduplication and retry timers to that session as well. Logout/account switching must clear user stores and the Query cache; a late response must not repopulate them.
- **Test behavior at risk boundaries.** Colocate focused `*.test.ts(x)` tests with the relevant domain module: unit tests for calculations/transitions, component/integration tests for API outcomes and user interactions. Reserve E2E tests for critical cross-route/browser flows under `e2e/` when introduced. Cover failure, retry, and session isolation where they matter; do not require a test per file or snapshots of static markup. Mock the network/API boundary rather than the behavior being verified.
- **Hold the baseline.** Run `npm run test`, `npx tsc --noEmit`, and `npm run lint` for behavioral changes; preserve existing lint warnings and follow the browser/build checks above. Keep the test environment independent of Node's native browser-storage implementation.
