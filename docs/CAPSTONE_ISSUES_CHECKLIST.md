# QuickPitik — Capstone Issues & Tasks Checklist

> **Tracking Document** for remaining bug fixes, feature enhancements, and architectural reconciliation tasks across Mobile, Web, Desktop, and Backend.

---

## 📋 Quick Status Overview

- **Total Tasks:** 31
- **✅ Already Solved:** 23 (Items 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 25, 26, 27, 28, 29, 31, 32, 33, 34)
- **⏳ Pending / To Be Worked On:** 8 (Items 16, 17, 19, 21, 22, 23, 24, 30)

---

## 🛠️ Detailed Checklist

### 📱 Mobile Application

- [x] **1. Photographer — Fix "Preview Public profile" button & Photo Preview**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `isFetchingBrandSettings` added to `PhotographerDashboardViewModel` to show a `CircularProgressIndicator` instead of an empty state when fetching. `PhotoPreviewMode.OwnerReview` passed explicitly in `PhotographerPublicProfileScreen` when previewed from studio mode (`cartViewModel == null`).

- [x] **2. Runner — Selfie Library "Gallery" button layout overflow (Mobile & Desktop)**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Refactored the mobile header layout (`ProfileScreen.kt`) to stack the Camera/Gallery buttons below the title using a `Column` instead of squeezing them in a `Row(SpaceBetween)`. On desktop web (`selfie-library.tsx`), added `md:grid-cols-4` to smooth out responsive upload tile scaling.

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

- [ ] **16. Mobile — Android 14 (API 34) USB Permission Broadcast Deadlock**
  - **Issue:** In `CameraConnectionManager.kt`, the broadcast receiver for `ACTION_USB_PERMISSION` (`com.quickpitik.mobile.USB_PERMISSION`) is registered using `ContextCompat.RECEIVER_NOT_EXPORTED`. When `UsbManager.requestPermission(device, pending)` is invoked, the permission dialog result is dispatched by Android's `system_server` (UID 1000). On Android 14+ (targetSdk 34), non-exported receivers reject broadcasts originating from outside the app UID for non-protected custom actions, which can cause camera permission callbacks to freeze or hang indefinitely.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/data/usb/CameraConnectionManager.kt`
  - **Action Plan:** Register the receiver with `ContextCompat.RECEIVER_EXPORTED` on API 34+ or use an explicit Intent/PendingIntent receiver targeted directly to the app package.

- [ ] **17. Mobile — In-Memory Buffering & Out-Of-Memory (OOM) Risk in Wi-Fi FTP**
  - **Issue:** `FtpReceiverServer.kt` reads incoming photo transfers completely into an in-memory `ByteArray`: `suspend (filename: String, bytes: ByteArray) -> Unit`. High-resolution DSLR/mirrorless RAW and JPEG images (25MB–50MB each) uploaded in rapid bursts over Wi-Fi can rapidly exhaust JVM heap memory and trigger an `OutOfMemoryError`.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/data/ftp/FtpReceiverServer.kt`
  - **Action Plan:** Stream incoming socket bytes directly to a temporary `File` via `OutputStream` (matching `UsbEventCaptureController.kt`) instead of buffering the entire image into RAM.

- [x] **18. Mobile — Orphaned MediaStore Rows & 0-Byte Placeholders on Camera Dismissal**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - In `ProfileScreen.kt`, updated `cameraLauncher` result callback to invoke `contentResolver.delete(uri, null, null)` and clear `tempImageUri` whenever `success == false`.
    - In `GalleryScreen.kt`, updated `selfieCameraLauncher` result callback to invoke `contentResolver.delete(uri, null, null)` and clear `pendingSelfieUri` whenever `success == false`.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/ProfileScreen.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/GalleryScreen.kt`

- [ ] **19. Mobile — Cross-User Upload Queue Leakage on Device Logout**
  - **Issue:** User sign-out (`AuthViewModel.logout()`) clears credentials via `sessionManager.clearSession()`, but the local SQLite `upload_queue` table is left unpurged. If a photographer logs out while frames are queued/failed and another photographer logs in on the same phone, `PhotoUploadWorker` will attempt to drain and upload Photographer A's photos under Photographer B's authentication token.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/auth/AuthViewModel.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/worker/PhotoUploadWorker.kt`
  - **Action Plan:** Clear or cancel pending upload queue items on logout, or enforce scoping by `photographerId` in `PhotoUploadWorker`.

- [x] **20. Mobile — Download Photo Filename Collisions on Multi-Photo Downloads**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Updated `PhotoDownloader.buildFilename()` to append `${photoId.take(8)}` to the bib slug (`quickpitik-bib-$bib-${photoId.take(8)}.jpg`).
    - Disambiguated multi-photo downloads for identical bib numbers across `OrdersScreen.kt`, `OrderReturnScreen.kt`, `GalleryScreen.kt`, and `PhotographerEventShareScreen.kt` while preserving human-readable bib labels.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/data/download/PhotoDownloader.kt`

- [ ] **21. Mobile — Deprecated Framework `android.media.ExifInterface` in Sync Worker**
  - **Issue:** `PhotoUploadWorker.kt` imports the legacy platform `android.media.ExifInterface` rather than AndroidX `androidx.exifinterface.media.ExifInterface`. The framework implementation has known orientation extraction bugs on older Android versions and can fail on vendor-specific camera EXIF metadata.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/worker/PhotoUploadWorker.kt`
  - **Action Plan:** Migrate import to `androidx.exifinterface.media.ExifInterface`.

- [ ] **22. Mobile — Compose Animation Performance & Non-Lambda Offset Overload**
  - **Issue:** `FloatingCart.kt` uses `.offset(x = pillOffsetX)` instead of the lambda overload `.offset { IntOffset(pillOffsetX.roundToPx(), 0) }` (flagged by Android Lint `UseOfNonLambdaOffsetOverload`). Non-lambda offset triggers recomposition of the entire composable during animation frames rather than executing purely in the layout phase, introducing UI jank during cart drags.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/FloatingCart.kt`
  - **Action Plan:** Switch to `Modifier.offset { IntOffset(...) }` lambda syntax.

- [ ] **23. Mobile — Material3 `Divider`, `LocalLifecycleOwner`, and State Autoboxing Modernization**
  - **Issue:**
    - Deprecated Material3 `Divider` used across `StudioShell.kt`, `GalleryScreen.kt`, `OrderReturnScreen.kt`, `PhotoPreview.kt`, `OrdersScreen.kt`, and `PhotographerEarningsScreen.kt` (should be `HorizontalDivider`).
    - Deprecated `androidx.compose.ui.platform.LocalLifecycleOwner` in `GalleryScreen.kt:166` and `EventsDiscoveryScreen.kt:110` (moved to `androidx.lifecycle.compose.LocalLifecycleOwner`).
    - Primitive autoboxing via `mutableStateOf(Int)` in `EventsDiscoveryScreen.kt`, `FloatingCart.kt`, `GalleryScreen.kt`, `OrdersScreen.kt`, and `PhotographerEarningsScreen.kt` (should be `mutableIntStateOf(Int)`).
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/*`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/photographer/*`
  - **Action Plan:** Update deprecated Material3 and Compose calls to current recommendations.

- [x] **31. Mobile & Web — Rebrand "Race Log" / "Race Logs" Terminology to "Saved Events"**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Updated navigation JumpSection and Slab heading in `website/src/app/profile/page.tsx` (`id: "saved-events"`, `label: "Saved events"`, `title="Saved events"`).
    - Updated toast redirect link in `website/src/components/events/save-button.tsx` to `#saved-events`.
    - Updated logout modal message in `website/src/app/account/page.tsx` to `"profile, saved events, and orders"`.
    - Updated runner profile kicker in `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/ProfileScreen.kt` to `Kicker("02 · Saved events")`.
    - Updated signout dialog in `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/AccountSettingsScreen.kt` to reference `"saved events"`.

- [x] **32. Mobile — Simplify Event Date Filter to 3 Options (Finished, Live, Upcoming)**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Consolidated `DateFilter` enum in `EventsDiscoveryScreen.kt` to 3 lifecycle options: `LIVE` ("Live"), `UPCOMING` ("Upcoming"), and `FINISHED` ("Finished", matching both `OPEN` and `PAST` event states), alongside `ANY` ("All events").
    - Updated matching predicate logic to filter accurately across single and grouped lifecycle states.

- [x] **33. Mobile — Hide "Live Now" Badge Once Actual Event Date Has Passed**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Updated `deriveEventState()` in `EventCatalog.kt` to restrict `EventState.LIVE` strictly to race day (`days == 0L`).
    - Decoupled photographer upload grace window (`canUploadToEvent()`, `days in 0L..3L`) from the runner-facing event state derivation, ensuring past events (`days > 0L`) transition directly to `OPEN` (Finished/Ready) with no lingering "Live now" badge.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/EventCatalog.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/EventsDiscoveryScreen.kt`
  - **Action Plan:** Restrict `EventState.LIVE` (or the "Live now" UI indicator) strictly to the actual day of the event (`days == 0L`), while preserving photographer upload capability window separately if needed.

- [x] **34. Mobile — Replace City Filtration with Philippine Regions Filtration**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Defined `PH_REGIONS` dataset and `resolveRegion(location: String)` in `EventCatalog.kt` covering all 17 administrative regions with keyword and province mappings matching the backend canonical regions.
    - Updated `EventsDiscoveryScreen.kt` to replace raw city extraction with region-based filtering (`regionFilter`, "All regions" dropdown, and matching predicate).
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/EventCatalog.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/EventsDiscoveryScreen.kt`

---

### 🌐 Website & Admin Panel

- [x] **6. Admin — Dispute store & queue wired to backend API**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `disputes-queue.tsx` uses `useAdminDisputes()` from `use-admin-data.ts` to hydrate live server data from `GET /api/v1/admin/disputes`. Admin resolve, deny, and escalate actions in `admin-dispute-store.ts` fire `apiResolveDispute()`, `apiDenyDispute()`, and `apiEscalateDispute()` calls in `api-admin.ts` backed by `AdminDisputeService.kt` and `PaymongoRefundService.kt`.

- [x] **7. Admin — Flag store & queue wired to backend API**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `AdminFlagService.kt` (transition guards, hide→HIDDEN cascade, dismiss/resolve of a hidden flag restores LIVE) + `AdminFlagsController.kt` (list/hide/dismiss/escalate/resolve, `q`). Website reads `GET /admin/flags` via `useAdminFlags()` (open + history pages) and mutates through `useFlagActions()` — server-authoritative, no optimistic override; failures surface as error toasts. Presigned thumbnails on `admin-flag-card.tsx`. Reviewed + hardened 2026-09-02 (`AdminFlagServiceTest`, V41 assertion in `FlywayMigrationIntegrationTest`). **No runner-side flag filing exists yet — the queue is empty until it lands.**

- [x] **8. Photographer — Settings hydrated from backend API**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** `api-photographer-settings.ts` provides backend API endpoints for brand (`putBrand`), handle (`putHandle`), region (`putRegion`), cover (`postCover`), watermark (`postWatermark`), socials, and payout accounts. `usePhotographerSettingsHydration()` mounted in `DashboardShell` populates store from `/me/photographer/brand` on login.

- [x] **13. Photographer — Google OAuth backend & onboarding flow**
  - **Status:** **ALREADY SOLVED IN CODE**
  - **Verification:** `GoogleAuthService.kt` (backend V38) implements `POST /auth/google` with auto-account linking, ID-token validation, and role onboarding selection (`ROLE_REQUIRED` 422). `google-button.tsx` uses Google Identity Services (GIS) and routes new users to `/onboarding`. Only requires configuring `NEXT_PUBLIC_GOOGLE_CLIENT_ID` in production environment.

- [ ] **24. Website — Runner-Side Photo Flagging / Report Flow (`AdminFlags` Parity Gap)**
  - **Issue:** The admin portal contains a complete flag moderation queue (`/admin/flags`), and the backend provides full moderation transition endpoints (`AdminFlagsController.kt`). However, there is no runner-facing UI (in `PhotoPreviewCard.tsx` or gallery mosaic tiles) to report/flag a photo, nor is there a runner-side API client or endpoint (`POST /api/v1/photos/{id}/flag`). The moderation queue therefore remains an unpopulated feature in production.
  - **Affected Files:**
    - `website/src/components/photos/photo-preview-card.tsx`
    - `website/src/lib/api-photos.ts`
    - `backend/src/main/kotlin/com/quickpitik/controller/AdminFlagsController.kt`
  - **Action Plan:** Add a "Report / Flag Photo" button and modal to `PhotoPreviewCard.tsx` allowing runners to submit reason categories (inappropriate content, privacy, incorrect bib tag), and connect to a backend flag submission endpoint.

- [x] **25. Website — Google OAuth Users Password Setting & Email Changes on `/account`**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Updated `UserDto.kt` and `UserDtoMapper.kt` in backend to expose `hasGoogleAuth = (user.googleSub != null)`.
    - Made `currentPassword` optional in `PasswordChangeRequest.kt` and `EmailChangeRequest.kt`.
    - Updated `ProfileService.kt` and `EmailChangeService.kt` to allow Google OAuth users to set their initial password or request an email change without entering a nonexistent legacy password, while continuing to mandate `currentPassword` for standard password accounts.
    - Updated `account/page.tsx` on website to show a Google account badge in the current email card, bypass current password requirement for Google OAuth users in `EmailSlab`, and render a dedicated "Set password" flow in `PasswordSlab`.
    - Added unit test suites in `backend/.../ProfileServiceTest.kt` and `EmailChangeServiceTest.kt` verifying password and email change flows for both Google OAuth and standard accounts.
  - **Affected Files:**
    - `backend/src/main/kotlin/com/quickpitik/dto/auth/UserDto.kt`
    - `backend/src/main/kotlin/com/quickpitik/service/profile/UserDtoMapper.kt`
    - `backend/src/main/kotlin/com/quickpitik/dto/profile/PasswordChangeRequest.kt`
    - `backend/src/main/kotlin/com/quickpitik/dto/profile/EmailChangeRequest.kt`
    - `backend/src/main/kotlin/com/quickpitik/service/profile/ProfileService.kt`
    - `backend/src/main/kotlin/com/quickpitik/service/profile/EmailChangeService.kt`
    - `backend/src/test/kotlin/com/quickpitik/service/profile/ProfileServiceTest.kt`
    - `backend/src/test/kotlin/com/quickpitik/service/profile/EmailChangeServiceTest.kt`
    - `website/src/types/user.ts`
    - `website/src/lib/api-account.ts`
    - `website/src/app/account/page.tsx`

- [x] **26. Website — Incomplete Base64URL Sanitization in `isTokenExpired()`**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Added `decodeBase64Url()` in `website/src/lib/auth.ts` to replace Base64URL characters (`-` $\rightarrow$ `+`, `_` $\rightarrow$ `/`) and apply standard padding (`==`, `=`).
    - Updated `isTokenExpired()` to decode JWT payloads through `decodeBase64Url()` without triggering DOMExceptions on valid active tokens.
    - Added unit test suite in `website/src/lib/auth.test.ts`.
  - **Affected Files:**
    - `website/src/lib/auth.ts`
    - `website/src/lib/auth.test.ts`

- [x] **27. Website — Missing App Routes in Photographer `RESERVED_HANDLES` Collision Guard**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Updated `RESERVED_HANDLES` set in `website/src/lib/reserved-handles.ts` to include missing core auth/verification routes: `"verify"`, `"verify-email"`, `"confirm-email-change"`, `"reset-password"`, `"logout"`, `"signin"`, and `"signup"`.
    - Created comprehensive unit test suite in `website/src/lib/reserved-handles.test.ts` verifying case-insensitive reserved handle rejection, valid handle rules, and required route collision protections.
  - **Affected Files:**
    - `website/src/lib/reserved-handles.ts`
    - `website/src/lib/reserved-handles.test.ts`

- [x] **28. Website — Floating-Point Precision Accumulation in Cart Pricing Summation**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Updated `total()` in `website/src/store/cart-store.ts` to accumulate item prices in integer cents (`Math.round(Number(item.price || 0) * 100)`) and return `cents / 100`.
    - Hardened `payable` calculation in `website/src/components/cart/checkout-modal.tsx` to round to integer cents before decimal conversion.
    - Added unit test suite in `website/src/store/cart-store.test.ts` verifying micro-decimal addition without floating-point drift.
  - **Affected Files:**
    - `website/src/store/cart-store.ts`
    - `website/src/components/cart/checkout-modal.tsx`
    - `website/src/store/cart-store.test.ts`

- [x] **29. Website — Overly Restrictive `Permissions-Policy` Blocking Camera in Production**
  - **Status:** **ALREADY SOLVED**
  - **Verification:**
    - Updated `Permissions-Policy` header in `website/next.config.ts` from `camera=()` to `camera=(self), microphone=(), geolocation=()`.
    - Enabled first-party selfie photo capture / runner selfie matching (`capture="user"` and web camera devices) while continuing to deny untrusted third-party iframe access.
  - **Affected Files:**
    - `website/next.config.ts`

- [ ] **30. Website — Sub-12px Quiet Studio Comfort Floor ESLint Warnings (121 instances)**
  - **Issue:** ESLint flags 121 warnings for text sizes below the 12px design token floor (`text-[9px]`, `text-[10px]`, `text-[11px]`) across components like `user-menu.tsx`, `onboarding-form.tsx`, `refund-timeline.tsx`, and `selfie-library.tsx`.
  - **Affected Files:**
    - `website/src/components/layout/user-menu.tsx`
    - `website/src/components/onboarding/onboarding-form.tsx`
    - `website/src/components/profile/selfie-library.tsx`
    - `website/src/components/orders/refund-timeline.tsx`
  - **Action Plan:** Replace raw sub-12px font utility classes with the design token `<Kicker>` component or normalize to `text-[12px]`.

---

### 📄 Documentation & Wireframes (SRS)

- [x] **15. SRS / Wireframe — Add updated Module 3 1.1 mobile wireframe**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Added `website/src/app/wireframes/srs/m3-1/page.tsx` rendering `UC-M3-1.1` Mobile Authentication & Verification wireframe. Registered route `UC-M3-1.1` in `website/src/app/wireframes/srs/page.tsx`.

