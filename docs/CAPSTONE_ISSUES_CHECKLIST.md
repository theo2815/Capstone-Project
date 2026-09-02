# QuickPitik — Capstone Issues & Tasks Checklist

> **Tracking Document** for remaining bug fixes, feature enhancements, and architectural reconciliation tasks across Mobile, Web, Desktop, and Backend.

---

## 📋 Quick Status Overview

- **Total Tasks:** 15
- **✅ Already Solved:** 8 (Items 1, 2, 7, 9, 10, 11, 12, 14)
- **⏳ Pending / To Be Worked On:** 7

---

## 🛠️ Detailed Checklist

### 📱 Mobile Application

- [x] **1. Photographer — Fix "Preview Public profile" button & Photo Preview**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `isFetchingBrandSettings` added to `PhotographerDashboardViewModel` to show a `CircularProgressIndicator` instead of an empty state when fetching. `PhotoPreviewMode.OwnerReview` passed explicitly in `PhotographerPublicProfileScreen` when previewed from studio mode (`cartViewModel == null`).

- [x] **2. Runner — Selfie Library "Gallery" button layout overflow (Mobile & Desktop)**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Refactored the mobile header layout (`ProfileScreen.kt`) to stack the Camera/Gallery buttons below the title using a `Column` instead of squeezing them in a `Row(SpaceBetween)`. On desktop web (`selfie-library.tsx`), added `md:grid-cols-4` to smooth out responsive upload tile scaling.

- [x] **7. Admin — Flag store & queue wired to backend API**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `AdminFlagService.kt` and `AdminFlagsController.kt` shipped on backend. `fetchAdminFlags()`, `resolveAdminFlag()`, `hideAdminFlag()`, `dismissAdminFlag()`, and `escalateAdminFlag()` wired in `api-admin.ts`. `useAdminFlags()` React Query hook integrated in `flags-queue.tsx` with optimistic store overrides in `admin-flag-store.ts`. Presigned S3 thumbnail previews rendered on `admin-flag-card.tsx` and aligned `FlagStatus` type with `"resolved"`.

- [x] **9. Photographer — Payout request on Mobile**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Implemented in `PhotographerEarningsScreen.kt` via `WalletSlab`, `OpenRequestBlock`, and `submitPayoutRequest()`.

- [x] **10. Runner — Email verification screen & deep linking on Mobile**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Added `intent-filter` for `/verify-email` in `AndroidManifest.xml` targeting `localhost` and `ngrok`. `MainActivity.kt` now routes `verify-email` Deep Links into the `NavHost`. Created `VerifyEmailScreen.kt` in Jetpack Compose to consume the token and submit it to `QuickPitikApi.verifyEmail()`.

- [x] **11. Runner — Confirm email change screen on Mobile**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Added `intent-filter` pathPrefix `/confirm-email-change` in `AndroidManifest.xml`. `MainActivity.kt` routes `/confirm-email-change` deep links into Compose `NavHost`. Created `ConfirmEmailChangeScreen.kt` to process the token via `QuickPitikApi.confirmEmailChange()`, clearing local session and prompting user to re-authenticate with their new email.

- [x] **12. Runner — Camera silent failure exception swallowed**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Replaced empty catch blocks with an `android.widget.Toast` in `ProfileScreen.kt` and `GalleryScreen.kt` so users receive immediate "Unable to open camera" feedback instead of silent failure.

- [x] **14. Runner — `GalleryScreen` `activeEvent!!` crash risk**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** All accesses to `activeEvent` in `GalleryScreen.kt` use null-safe calls (`activeEvent?.name`) and null checks (`val event = activeEvent; if (event != null)`). No `!!` unwraps exist.

---

### 💻 Desktop Application (BatchMyPhotos)

- [ ] **3. Desktop — Settings Page Missing**
  - **Issue:** The Electron desktop application lacks a dedicated Settings page for photographer profile/preferences.
  - **Target Location:** `BatchMyPhotos` (External desktop repo)
  - **Action Plan:** Build Settings view in the Electron app allowing photographers to view/manage their account credentials and app configs.

- [ ] **4. Desktop — Invalid Role on Registration**
  - **Issue:** Backend `RegisterRequest.kt` expects strict enum `Role` (`ADMIN`, `PHOTOGRAPHER`, `RUNNER`). Desktop client registration fails if sending lowercase or mismatched role values.
  - **Affected Files:**
    - `BatchMyPhotos` (Desktop auth registration service)
    - `backend/src/main/kotlin/com/quickpitik/dto/auth/RegisterRequest.kt`
  - **Action Plan:** Ensure Desktop app sends `PHOTOGRAPHER` as uppercase string matching backend enum.

- [ ] **5. Desktop — Invalid Reset Password Token**
  - **Issue:** Backend V37 uses a 2-step OTP flow (`/verify-reset-otp` → 15-min continuation token → `/reset-password`). Desktop sends tokens directly to `reset-password` without obtaining the continuation token.
  - **Affected Files:**
    - `BatchMyPhotos` (Desktop password reset flow)
  - **Action Plan:** Update Desktop app to implement the 2-step OTP verification flow matching web/mobile.

---

### 🌐 Website & Admin Panel

- [ ] **6. Admin — Dispute store is mock-only**
  - **Issue:** `admin-dispute-store.ts` uses local state and `ADMIN_DISPUTES = []` rather than querying `GET /api/v1/admin/disputes`.
  - **Affected Files:**
    - `website/src/store/admin-dispute-store.ts`
    - `website/src/lib/admin-disputes.ts`
    - `website/src/lib/admin-dispute-view.ts`
    - `website/src/lib/api-admin.ts`
  - **Action Plan:** Wire real query hydration from `fetchAdminDisputes()` in `api-admin.ts` to populate the dispute queue with live server data.

- [ ] **8. Photographer — `photographer-settings-store` uses `localStorage`**
  - **Issue:** Zustand store persists state in `localStorage` via `persist()`, which can go stale or exceed localStorage limits with base64/data URLs.
  - **Affected Files:**
    - `website/src/store/photographer-settings-store.ts`
    - `website/src/hooks/use-photographer-settings-hydration.ts`
  - **Action Plan:** Transition from local `persist` middleware to server-authoritative React Query / cache fetching.

- [ ] **13. Photographer — Google OAuth `OAUTH_ENABLED = false`**
  - **Issue:** `google-button.tsx` disables Google sign-in when `NEXT_PUBLIC_GOOGLE_CLIENT_ID` is empty. Also, backend Google OAuth currently defaults new accounts to `RUNNER`.
  - **Affected Files:**
    - `website/src/components/auth/google-button.tsx`
    - `website/src/lib/constants.ts`
    - `backend/src/main/kotlin/com/quickpitik/service/auth/GoogleAuthService.kt`
  - **Action Plan:** Supply Google Cloud client ID configuration and handle role selection/onboarding for photographers signing up via Google.

---

### 📄 Documentation & Wireframes (SRS)

- [ ] **15. SRS / Wireframe — Add updated Module 3 1.1 mobile wireframe**
  - **Issue:** Module 3 currently only has Web wireframes (`m3-3` through `m3-8`) and a text placeholder for M3.1 auth in `SRS-QuickPitik.md`.
  - **Affected Files:**
    - `website/src/app/wireframes/srs/`
    - `website/src/app/wireframes/srs/page.tsx`
    - `Papers-For-Capstone/SRS-QuickPitik.md`
  - **Action Plan:** Add the mobile wireframe page for Module 3 1.1 and update the SRS document links.
