# QuickPitik — Capstone Issues & Tasks Checklist

> **Tracking Document** for remaining bug fixes, feature enhancements, and architectural reconciliation tasks across Mobile, Web, Desktop, and Backend.

---

## 📋 Quick Status Overview

- **Total Tasks:** 15
- **✅ Already Solved:** 3 (Items 1, 9, 14)
- **⏳ Pending / To Be Worked On:** 12

---

## 🛠️ Detailed Checklist

### 📱 Mobile Application

- [x] **1. Photographer — Fix "Preview Public profile" button & Photo Preview**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `isFetchingBrandSettings` added to `PhotographerDashboardViewModel` to show a `CircularProgressIndicator` instead of an empty state when fetching. `PhotoPreviewMode.OwnerReview` passed explicitly in `PhotographerPublicProfileScreen` when previewed from studio mode (`cartViewModel == null`).

- [ ] **2. Runner — Selfie Library "Gallery" button layout overflow (Mobile & Desktop)**
  - **Issue:** The header in `ProfileScreen.kt` squashes the *Camera* and *Gallery* pill buttons on narrower viewports. On desktop web, upload tile sizing needs responsive cleanup.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/ProfileScreen.kt`
    - `website/src/components/profile/selfie-library.tsx`
  - **Action Plan:** Refactor the mobile header layout to stack or use a responsive button group so buttons never clip. Polish web upload tile grid breakpoints.

- [x] **9. Photographer — Payout request on Mobile**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Implemented in `PhotographerEarningsScreen.kt` via `WalletSlab`, `OpenRequestBlock`, and `submitPayoutRequest()`.

- [ ] **10. Runner — Email verification screen & deep linking on Mobile**
  - **Issue:** Registration email link opens `$frontendOrigin/verify-email?token=...` in the browser. Mobile has no deep link handler or native `VerifyEmailScreen`.
  - **Affected Files:**
    - `mobile/app/src/main/AndroidManifest.xml`
    - `mobile/app/src/main/java/com/quickpitik/mobile/MainActivity.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/auth/VerifyEmailScreen.kt` (New)
    - `backend/src/main/kotlin/com/quickpitik/service/EmailService.kt`
  - **Action Plan:** Register deep link scheme for verification and create mobile verification screen or universal link handling.

- [ ] **11. Runner — Confirm email change screen on Mobile**
  - **Issue:** Email change initiated from mobile sends confirmation email linking to the web app (`/confirm-email-change`). No mobile UI completes step 2.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/data/remote/QuickPitikApi.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/ProfileScreen.kt`
    - `backend/src/main/kotlin/com/quickpitik/service/EmailService.kt`
  - **Action Plan:** Add mobile route / dialog to handle confirmation token or deep-link into the app.

- [ ] **12. Runner — Camera silent failure exception swallowed**
  - **Issue:** Exceptions during camera launch / MediaStore URI insertion in `ProfileScreen.kt` and `GalleryScreen.kt` are caught in empty catch blocks without user feedback.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/ProfileScreen.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/GalleryScreen.kt`
  - **Action Plan:** Add user-facing feedback (Toast / Snackbar / inline error state) when camera launch fails (e.g., camera permission denied or storage failure).

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

- [ ] **7. Admin — Flag store is mock-only**
  - **Issue:** `admin-flag-store.ts` stores moderation flags purely in `localStorage` without backend integration.
  - **Affected Files:**
    - `website/src/store/admin-flag-store.ts`
    - `backend` (Admin Flag API endpoints if required by SRS)
  - **Action Plan:** Connect flag store to backend moderation endpoints or formally document mock status for demo scope.

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
