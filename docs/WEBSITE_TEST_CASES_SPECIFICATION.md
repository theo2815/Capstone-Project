# QuickPitik Website — Automated Test Cases Specification & Testing Strategy

> **Status:** Proposed & Approved for Implementation  
> **Scope:** `website/` (Next.js 15, React 19, TypeScript, Zustand, TanStack Query, Tailwind CSS)  
> **Target Tooling:** Vitest + React Testing Library + jsdom + MSW (Mock Service Worker)

---

## 1. Executive Overview & Testing Objectives

The QuickPitik web application coordinates critical user journeys across runners, photographers, and administrators:
- **Runners:** Photo discovery (bib search, facial recognition selfie match), shopping cart management, PayMongo hosted checkout, order receipt polling, and high-resolution photo downloads.
- **Photographers:** Direct batch photo uploads with client-side SHA-256 deduplication and 4-slot concurrency throttling, brand profile customization, earnings breakdown, and payout requests.
- **Administrators:** Dispute queue resolution, content flags moderation, photographer verifications, and platform analytics.

Currently, the `website/` codebase has **0 automated unit, integration, or component tests**. To prevent regressions, ensure compliance with capstone requirements, and validate recent diagnostic findings, this document defines the testing architecture, proposed framework, and comprehensive suite of test cases to be implemented.

---

## 2. Test Architecture & Tooling Selection

### 2.1 Recommended Test Runner: Vitest
We recommend **Vitest** over Jest for the following reasons:
1. **First-Class Next.js 15 & React 19 Compatibility:** Vitest natively supports modern ESM, React 19 JSX transforms, and TypeScript without heavy Babel or SWC transpilation shims.
2. **Speed & Efficiency:** Instant startup via Vite's esbuild transform, blazing fast execution across hundreds of tests.
3. **Familiar Jest-Compatible API:** Uses `describe`, `it`, `expect`, `vi.fn()`, and `vi.spyOn()`, making it seamless for developers familiar with Jest.
4. **Clean DOM Simulation:** Integrates cleanly with `jsdom` and `@testing-library/react`.

### 2.2 Testing Stack Components
| Tool / Package | Version Target | Purpose |
| :--- | :--- | :--- |
| `vitest` | `^3.x` | Test runner, test orchestration, assertions, and mock functions |
| `@testing-library/react` | `^16.x` | Component rendering, event dispatching, and user interaction simulation |
| `@testing-library/jest-dom` | `^6.x` | Custom matchers for DOM assertions (`toBeInTheDocument`, `toBeDisabled`) |
| `@testing-library/user-event` | `^14.x` | Realistic browser event simulations (typing, clicking, file uploads) |
| `jsdom` | `^26.x` | Browser environment simulation (Window, Document, localStorage, Event) |

### 2.3 Proposed Directory Structure
All test files will live alongside their implementation or in dedicated `__tests__` directories:
```text
website/
├── src/
│   ├── lib/
│   │   ├── __tests__/
│   │   │   ├── auth.test.ts
│   │   │   ├── auth-validation.test.ts
│   │   │   ├── reserved-handles.test.ts
│   │   │   ├── image-utils.test.ts
│   │   │   └── api-client.test.ts
│   ├── store/
│   │   ├── __tests__/
│   │   │   ├── cart-store.test.ts
│   │   │   ├── auth-store.test.ts
│   │   │   ├── saved-events-store.test.ts
│   │   │   └── admin-dispute-store.test.ts
│   ├── hooks/
│   │   ├── __tests__/
│   │   │   ├── use-auth.test.ts
│   │   │   ├── use-url-state.test.ts
│   │   │   └── use-runner-notifications-ws.test.ts
│   ├── components/
│   │   ├── cart/__tests__/
│   │   │   ├── floating-cart.test.tsx
│   │   │   └── checkout-modal.test.tsx
│   │   ├── photos/__tests__/
│   │   │   └── photo-preview-card.test.tsx
│   │   └── events/__tests__/
│   │       └── bib-search-panels.test.tsx
│   └── app/
│       └── orders/return/__tests__/
│           └── return-polling.test.tsx
├── vitest.config.mts
└── vitest.setup.ts
```

---

## 3. Elaborated Test Suites & Detailed Test Cases

---

### Suite 1: Authentication & Token Lifecycle (`src/lib/auth.ts`, `src/lib/api.ts`, `src/hooks/use-auth.ts`)
*Focuses on security, session resilience, single-flight token rotation, and proper JWT parsing.*

- **Case 1.1: JWT Expiration Detection (`isTokenExpired`)**
  - **Scenario A:** Standard base64 payload returns `false` when `exp` is in the future.
  - **Scenario B:** Expired payload returns `true` when `exp * 1000 < Date.now()`.
  - **Scenario C (Critical Regression Test):** URL-safe Base64URL characters (`-` and `_`) in the JWT payload must NOT cause `atob()` to throw `DOMException`; must cleanly parse without premature expiration (verifying Item 26 from the Capstone Checklist).
  - **Scenario D:** Malformed/garbage token string returns `true` gracefully instead of uncaught crash.

- **Case 1.2: Token Storage & Accessors**
  - Verify `setTokens()`, `getAccessToken()`, `getRefreshToken()`, and `clearTokens()` interact correctly with browser `localStorage`.
  - Verify safe fallback to `null` when executed in a non-browser environment (`typeof window === "undefined"`).

- **Case 1.3: Single-Flight Refresh Token Rotation (`doRefresh` in `api.ts`)**
  - **Scenario A:** When 3 concurrent HTTP requests encounter a `401 Unauthorized`, only **one** refresh request (`POST /auth/refresh`) is dispatched to the backend.
  - **Scenario B:** Once the single-flight refresh resolves, all 3 original requests retry with the freshly minted access token and resolve successfully.
  - **Scenario C:** If the refresh request itself fails with `401/403`, all pending requests reject and the user session is purged via `redirectToLogin()`.

- **Case 1.4: Session Wipe & Guest Buffer Carry-over (`useAuth.establishSession`)**
  - Verify `establishSession()` clears previous user React Query caches and user-scoped stores, while carrying over the guest shopping cart items and saved event bookmarks (`captureGuestBuffer()`).

---

### Suite 2: Shopping Cart & Pricing State (`src/store/cart-store.ts`, `src/components/cart/`)
*Validates cart mutations, maximum caps, price aggregation, and guest-to-server synchronization.*

- **Case 2.1: Item Addition & Deduplication**
  - Adding a photo item appends it to `items`.
  - Adding an already present `photoId` is ignored (idempotent add).
  - Verifying `addItem()` returns `true` on success and `false` when ignored.

- **Case 2.2: Hard Capacity Cap (`MAX_CART_ITEMS = 100`)**
  - When the cart reaches 100 items, `isFull()` returns `true`.
  - Attempting to add the 101st item is rejected (`addItem()` returns `false`, `items.length` remains 100).
  - `remainingCapacity()` accurately tracks available slots (`100 - items.length`).

- **Case 2.3: Currency Precision Total Summation**
  - When adding multiple items with fractional cent values (e.g., ₱150.50 + ₱99.99 + ₱49.51), `total()` must equal `300.00` without IEEE-754 floating-point drift (`300.00000000000006`).
  - Verifies resolution of Item 28 from the checklist.

- **Case 2.4: Optimistic Server Sync & Rollback**
  - When `syncEnabled == true`, calling `addItem()` optimistically updates state and calls `postCartItem()`.
  - If `postCartItem()` succeeds and returns a canonical server price, local row price updates to match the server.
  - If `postCartItem()` rejects with network error, the item is rolled back from local state.
  - Symmetrical behavior for `removeItem()` and `deleteCartItem()`.

---

### Suite 3: Validation Utilities & Security Guards (`src/lib/`)
*Validates input sanitize rules, regex guards, and namespace reservations.*

- **Case 3.1: Reserved Photographer Handles (`src/lib/reserved-handles.ts`)**
  - Verify core reserved slugs are blocked: `"admin"`, `"dashboard"`, `"login"`, `"register"`, `"api"`, `"cart"`.
  - **Regression Test:** Verify `"verify"`, `"verify-email"`, and `"confirm-email-change"` are strictly rejected as reserved handles (Item 27).
  - Valid custom handles (e.g., `"john-lens"`, `"marathon-pix"`) are accepted.
  - Invalid formats (leading dash, trailing dash, consecutive dashes, symbols, uppercase, <3 or >32 chars) return descriptive error messages.

- **Case 3.2: Auth Validation Rules (`src/lib/auth-validation.ts`)**
  - Email regex rejects missing `@`, invalid TLDs, and leading/trailing whitespace.
  - Password validator enforces minimum length (8 chars), rejecting empty or short inputs.
  - OTP validator ensures strictly 6-digit numeric codes (`^\d{6}$`).

- **Case 3.3: Image File Constraints (`src/lib/image-utils.ts`)**
  - Rejects files exceeding `MAX_UPLOAD_BYTES` (10 MB).
  - Accurately detects HEIC/HEIF files (`isHeicFile`) by MIME type and filename extension.
  - Validates accepted MIME types (`image/jpeg`, `image/png`, `image/webp`).

---

### Suite 4: Photo Discovery, Bib Search & Lightbox (`src/components/`)
*Validates runner search experience and modal interactions.*

- **Case 4.1: Bib Search Panel (`src/components/events/bib-search-panels.tsx`)**
  - Form submission with valid bib number triggers `onBibChange` and dispatches search query.
  - Form trims whitespace and handles alphanumeric bib tags (e.g. `"B-4082"`).
  - Mode toggle switches between Bib mode and Selfie mode.

- **Case 4.2: Photo Lightbox Preview Card (`src/components/photos/photo-preview-card.tsx`)**
  - **Mode Switching:**
    - In `"browse"` mode: displays "Buy now" / "Add to cart" buttons and watermark warning.
    - In `"owned"` mode: displays "Download photo" button and unwatermarked `cleanUrl`.
    - In `"review"` mode: renders read-only admin judgment layout without commerce CTAs.
  - **Keyboard Navigation:** Pressing `Escape` calls `onClose()`, `ArrowLeft` calls `onPrev()`, `ArrowRight` calls `onNext()`.
  - **Photographer Attribution:** Formats verified photographer handle as a link to `/{handle}`; unverified photographers display plain name without hyperlink.

---

### Suite 5: Order Completion & Return Polling (`src/app/orders/return/page.tsx`)
*Validates PayMongo redirect landing, status polling, and state transitions.*

- **Case 5.1: Polling Happy Path**
  - Renders `<PollingState>` while `status === "PENDING"`.
  - Once backend returns `status === "PAID"` or `status === "FULFILLED"`:
    - Clears runner cart (`clearCart()`).
    - Invalidates query cache (`["me", "orders"]`).
    - Transitions to `<PaidState>` rendering receipt and photo cards.

- **Case 5.2: Polling Timeout Handling**
  - If polling exceeds `POLL_MAX_ATTEMPTS` (30 attempts / 60 seconds) without a terminal status, smoothly transitions to `<TimeoutState>` displaying order reference and retry button.

- **Case 5.3: Error & Cancellation Handling**
  - If order status returns `REFUNDED` or endpoint returns 404, transitions to `<FailedState>`.

---

### Suite 6: Photographer Direct Upload Flow (`src/lib/api-photographer.ts`)
*Validates batch upload concurrency control, hash deduplication, and auth token rotation.*

- **Case 6.1: Concurrency Throttling**
  - Given a queue of 20 photos, only `MAX_CONCURRENT_UPLOADS` (4) active XHR requests are processed simultaneously.
  - When one upload completes, the next queued item immediately occupies the slot.

- **Case 6.2: Mid-Batch 401 Access Token Expiry Handling**
  - During a large 500-photo batch where the 15-minute access token expires, an upload encountering `401 Unauthorized` invokes `refreshAccessToken()` once and retries without dropping the batch.

- **Case 6.3: Pre-Flight Duplicate Checking**
  - `checkPhotosExist()` hashes files locally via SHA-256 and skips uploading files flagged as `already_in_event`.

---

## 4. Implementation Steps & Milestones

1. **Milestone 1 — Setup Test Infrastructure:**
   - Install `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`, and `jsdom` as `devDependencies`.
   - Create `vitest.config.mts` configured with path alias `@/*` -> `./src/*` and setup file.
   - Add `"test": "vitest run"`, `"test:watch": "vitest"`, and `"test:coverage": "vitest run --coverage"` to `website/package.json`.
   - Execute baseline command to confirm 0 failures on an empty suite.

2. **Milestone 2 — Utility & Store Unit Tests (Suites 1, 2, 3):**
   - Implement `auth.test.ts` (including Base64URL decoding test).
   - Implement `cart-store.test.ts` (capacity caps, deduplication, price summation).
   - Implement `reserved-handles.test.ts` and `auth-validation.test.ts`.

3. **Milestone 3 — Component & Lifecycle Tests (Suites 4, 5, 6):**
   - Implement `bib-search-panels.test.tsx` and `photo-preview-card.test.tsx`.
   - Implement order return polling test `return-polling.test.tsx`.
   - Implement upload queue concurrency logic test `api-photographer.test.ts`.

4. **Milestone 4 — CI & Checklist Verification:**
   - Run complete test suite and record results.
   - Update `CAPSTONE_ISSUES_CHECKLIST.md` as testing safeguards are established.
