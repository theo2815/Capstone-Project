# QuickPitik — Capstone Issues & Tasks Checklist

> **Tracking Document** for remaining bug fixes, feature enhancements, and architectural reconciliation tasks across Mobile, Web, Desktop, and Backend.

---

## 📋 Quick Status Overview

- **Total Tasks:** 15
- **✅ Already Solved:** 11 (Items 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14)
- **⏳ Pending / To Be Worked On:** 4

---

## 🛠️ Detailed Checklist

### 📱 Mobile Application

- [x] **1. Photographer — Fix "Preview Public profile" button & Photo Preview**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `isFetchingBrandSettings` added to `PhotographerDashboardViewModel` to show a `CircularProgressIndicator` instead of an empty state when fetching. `PhotoPreviewMode.OwnerReview` passed explicitly in `PhotographerPublicProfileScreen` when previewed from studio mode (`cartViewModel == null`).

- [x] **2. Runner — Selfie Library "Gallery" button layout overflow (Mobile & Desktop)**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Refactored the mobile header layout (`ProfileScreen.kt`) to stack the Camera/Gallery buttons below the title using a `Column` instead of squeezing them in a `Row(SpaceBetween)`. On desktop web (`selfie-library.tsx`), added `md:grid-cols-4` to smooth out responsive upload tile scaling.

- [x] **6. Admin — Dispute store & queue wired to backend API**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `disputes-queue.tsx` uses `useAdminDisputes()` from `use-admin-data.ts` to hydrate live server data from `GET /api/v1/admin/disputes`. Admin resolve, deny, and escalate actions in `admin-dispute-store.ts` fire `apiResolveDispute()`, `apiDenyDispute()`, and `apiEscalateDispute()` calls in `api-admin.ts` backed by `AdminDisputeService.kt` and `PaymongoRefundService.kt`.

- [x] **7. Admin — Flag store & queue wired to backend API**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `AdminFlagService.kt` and `AdminFlagsController.kt` shipped on backend. `fetchAdminFlags()`, `resolveAdminFlag()`, `hideAdminFlag()`, `dismissAdminFlag()`, and `escalateAdminFlag()` wired in `api-admin.ts`. `useAdminFlags()` React Query hook integrated in `flags-queue.tsx` with optimistic store overrides in `admin-flag-store.ts`. Presigned S3 thumbnail previews rendered on `admin-flag-card.tsx` and aligned `FlagStatus` type with `"resolved"`.

- [x] **8. Photographer — Settings hydrated from backend API**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `api-photographer-settings.ts` provides backend API endpoints for brand (`putBrand`), handle (`putHandle`), region (`putRegion`), cover (`postCover`), watermark (`postWatermark`), socials, and payout accounts. `usePhotographerSettingsHydration()` mounted in `DashboardShell` populates store from `/me/photographer/brand` on login.

- [x] **13. Photographer — Google OAuth backend & onboarding flow**
  - **Status:** **ALREADY SOLVED IN CODE**
  - **Verification:** `GoogleAuthService.kt` (backend V38) implements `POST /auth/google` with auto-account linking, ID-token validation, and role onboarding selection (`ROLE_REQUIRED` 422). `google-button.tsx` uses Google Identity Services (GIS) and routes new users to `/onboarding`. Only requires configuring `NEXT_PUBLIC_GOOGLE_CLIENT_ID` in production environment.

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

### 📄 Documentation & Wireframes (SRS)

- [ ] **15. SRS / Wireframe — Add updated Module 3 1.1 mobile wireframe**
  - **Issue:** Module 3 currently only has Web wireframes (`m3-3` through `m3-8`) and a text placeholder for M3.1 auth in `SRS-QuickPitik.md`.
  - **Affected Files:**
    - `website/src/app/wireframes/srs/`
    - `website/src/app/wireframes/srs/page.tsx`
    - `Papers-For-Capstone/SRS-QuickPitik.md`
  - **Action Plan:** Add the mobile wireframe page for Module 3 1.1 and update the SRS document links.
