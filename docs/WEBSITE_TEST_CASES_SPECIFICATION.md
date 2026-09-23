# QuickPitik Website - Automated Test Cases Specification & Testing Strategy

> **Status:** Implemented; architecture workstream closed with acceptance follow-ups tracked separately.
> **Scope:** `website/` automated unit, component and mocked API-boundary regression tests.
> **Recorded baseline:** 391 passing tests across 46 test files, measured on 2026-09-12 at application commit `ba239c7`.
> **Candidate check:** 438 passing tests across 48 test files, measured on 2026-09-20 at `c90736ee` with Node 22.23.2.

## 1. Objectives and Evidence

The suite protects authentication/session isolation, server-data ownership, administrative mutations, photographer workflows, and runner checkout behavior. Phase 2 moved server records into TanStack Query; Phase 3 separated Settings queries from editable drafts and Checkout orchestration from presentation.

The baseline is a historical execution result, not a claim that every product journey is covered. Test files and `website/package.json` are the live source of truth; rerun the command below for current totals.

| Recorded check | Result on 2026-09-12 |
| --- | --- |
| `npm run test` | 391 tests passed in 46 files |
| `npx tsc --noEmit` | Passed |
| `npm run lint` | 0 errors; 118 existing warnings |
| `NEXT_PUBLIC_API_URL=http://localhost:8080/api/v1 npm run build` | Passed |

The candidate's 2026-09-20 `npm run test` check passed 438 tests in 48 files. Lint, typechecking, builds and manual browser checks are separate; no code-coverage percentage is claimed.

## 2. Implemented Test Architecture

| Concern | Implementation / source of truth |
| --- | --- |
| Runner and assertions | Vitest; dependency ranges in `website/package.json` |
| Component interaction | React Testing Library, jest-dom and user-event |
| DOM environment | jsdom |
| Configuration | `website/vitest.config.ts`: automatic JSX, `@` alias to `src`, jsdom environment |
| Shared setup | `website/src/test/setup.ts` |
| Test placement | Colocated `.test.ts` and `.test.tsx` files under `website/src/` |
| API isolation | Mocked dependencies/API responses; does not exercise a live Spring Boot or payment service |

MSW was proposed in the original plan but is not a dependency in the current package manifest. Dedicated `test:watch` and `test:coverage` scripts are not configured; do not treat them as shipped tooling.

Run from the website directory:

```sh
cd website
npm run test
```

## 3. Automated Test File Inventory

Paths below are relative to `website/src/`. This inventory identifies the 48 test files at candidate `c90736ee`. Individual assertions, parameterized cases and expected outcomes remain beside the implementation.

| Area | Test files |
| --- | --- |
| Session and cart stores | `store/guest-buffer-session.test.ts`, `store/cart-store.test.ts` |
| Authentication and session utilities | `lib/auth.test.ts`, `lib/auth-reset.test.ts`, `lib/api-session.test.ts` |
| Photographer and catalog utilities | `lib/api-photographer.test.ts`, `lib/photographer-verification-sync.test.ts`, `lib/event-catalog.test.ts` |
| Validation, media and browser utilities | `lib/reserved-handles.test.ts`, `lib/utils.test.ts`, `lib/watermark-placement.test.ts`, `lib/save-photo.test.ts`, `lib/in-app-browser.test.ts` |
| Query ownership and admin hooks | `hooks/private-query-boundaries.test.tsx`, `hooks/admin-query-mutations.test.tsx`, `hooks/use-admin-events.test.tsx`, `hooks/use-admin-users.test.tsx`, `hooks/use-admin-payouts.test.tsx` |
| Notifications and inbox hooks | `hooks/use-inbox-messages.test.tsx`, `hooks/use-inbox-notifications-ws.test.tsx`, `hooks/use-admin-notifications-ws.test.tsx` |
| Photographer hooks | `hooks/use-can-upload.test.ts`, `hooks/use-photographer-payout-actions.test.tsx`, `hooks/use-photographer-settings-hydration.test.tsx` |
| Checkout hooks | `hooks/use-checkout-actions.test.tsx`, `hooks/use-checkout-workflow.test.tsx` |
| Auth, account and layout components | `components/auth/payout-security.test.tsx`, `components/auth/google-button.test.tsx`, `components/account/avatar-slab.test.tsx`, `components/layout/auth-hydrator.test.tsx`, `components/layout/notification-inboxes.test.tsx` |
| Photographer components | `components/dashboard/watermark-placement-editor.test.tsx`, `components/dashboard/event-coupon-modal.test.tsx`, `components/dashboard/file-payout-report-modal.test.tsx` |
| Admin components | `components/admin/admin-user-actions.test.tsx`, `components/admin/event-requests-queue.test.tsx`, `components/admin/payouts-queue.test.tsx`, `components/admin/admin-mutation-flows.test.tsx` |
| Photo and checkout components | `components/photos/photo-preview-card.test.tsx`, `components/events/photo-mosaic-tile.test.tsx`, `components/events/bib-search-panels.test.tsx`, `components/profile/selfie-library.test.tsx`, `components/cart/checkout-modal.test.tsx`, `components/cart/pending-payment-watcher.test.tsx` |
| Route-level workflows | `app/admin/disputes/[id]/page.test.tsx`, `app/admin/events/page.test.tsx`, `app/dashboard/settings/page.test.tsx`, `app/dashboard/events/photo-management.test.tsx` |

## 4. Architecture Regression Coverage

| Workstream | Covered regression themes | Principal test homes |
| --- | --- | --- |
| Session isolation | Guest-buffer ownership, auth reset/hydration, stale asynchronous results and private query boundaries | Store/session tests, `auth-hydrator.test.tsx`, `private-query-boundaries.test.tsx` |
| Server-data ownership | Query/mutation behavior, admin events/users/payouts, inbox data and WebSocket invalidation | Admin query and inbox hook tests; admin component/route tests |
| Settings separation | Query hydration and editable Settings workflow | `use-photographer-settings-hydration.test.tsx`, Settings `page.test.tsx` |
| Checkout separation | Pending-payment recovery, QR creation, paid receipt snapshots, zero-total settlement, retry/idempotency, cancellation outcomes, dead capabilities and timer cleanup | `use-checkout-workflow.test.tsx`, `use-checkout-actions.test.tsx`, Checkout component tests |
| Checkout review fixes | Late create/status/cancel failures after a session switch do not affect the new session; QR and receipt totals use the server-confirmed order amount, including zero-total settlement | `use-checkout-workflow.test.tsx` |

The original six-suite proposal is superseded by this implemented inventory. Proposed standalone auth-validation, image-utils, bib-search and order-return polling test files are not listed as delivered. Existing related tests do not establish complete coverage of those earlier proposed scenarios.

## 5. Browser Evidence and Acceptance Limits

Separate local Chrome checks used fictional API responses, not real provider transactions:

| Surface | Recorded local browser evidence |
| --- | --- |
| Settings | Save failure retains draft, retry succeeds, refresh restores saved data; responsive checks at 375/768/1024 CSS pixels |
| Guest Checkout | Email confirmation, failed creation/retry, QR refresh recovery, cancellation failure/retry, server-confirmed amount, simulated settlement and receipt layout |
| Signed-in Checkout | Authenticated status polling, QR recovery, simulated settlement, Orders cache refresh, receipt persistence and cart cleanup |

These manual checks are not automated end-to-end tests and are not part of the automated test counts. Viewport resizing does not establish physical-device or native webview behavior.

## 6. Closeout and Remaining Acceptance

The architecture implementation and its local regression workstream are closed at the user's request. Closure does not assert production readiness, merge into the default branch, or completion of every website backlog item.

| Follow-up | Status |
| --- | --- |
| Real PayMongo sandbox payment and webhook confirmation | Deferred at the user's request; unverified |
| Actual receipt email and full-resolution/signed downloads | Open; not established by fixture receipt checks |
| Physical Android/iOS browsers and FB/Instagram/Messenger webviews | Evidence remains incomplete; S9 is accepted and excluded from the TASK 11 release gate. No physical-device acceptance is claimed. |
| Additional proposed test scenarios and coverage reporting | Not claimed by this baseline; assess separately when those surfaces change |

Live ownership and follow-up scheduling belong in the Second Brain `website/tasks.md`; implementation evidence and closeout rationale are in `_journal/2026-09-11-phase-3-settings.md`. Keep this specification as the versioned testing reference, not a second live task board.
