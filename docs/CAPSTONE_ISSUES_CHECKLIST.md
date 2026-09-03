# QuickPitik — Capstone Issues & Tasks Checklist

> **Tracking Document** for remaining bug fixes, feature enhancements, and architectural reconciliation tasks across Mobile, Web, Desktop, and Backend.

---

## 📋 Quick Status Overview

- **Total Tasks:** 20
- **✅ Already Solved:** 12 (Items 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
- **⏳ Pending / To Be Worked On:** 8 (Items 16, 17, 18, 19, 20, 21, 22, 23)

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

- [ ] **18. Mobile — Orphaned MediaStore Rows & 0-Byte Placeholders on Camera Dismissal**
  - **Issue:** A MediaStore entry is inserted into `MediaStore.Images.Media.EXTERNAL_CONTENT_URI` prior to launching the system camera contract `ActivityResultContracts.TakePicture()`. When the user opens the camera and cancels (or presses back without capturing a photo), `success == false`. Because no cleanup runs on cancellation, an empty 0-byte corrupt image placeholder is permanently orphaned in the user's public device gallery.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/ProfileScreen.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/runner/GalleryScreen.kt`
  - **Action Plan:** In the `TakePicture()` result callback, delete the inserted URI via `contentResolver.delete(uri, null, null)` whenever `success == false`.

- [ ] **19. Mobile — Cross-User Upload Queue Leakage on Device Logout**
  - **Issue:** User sign-out (`AuthViewModel.logout()`) clears credentials via `sessionManager.clearSession()`, but the local SQLite `upload_queue` table is left unpurged. If a photographer logs out while frames are queued/failed and another photographer logs in on the same phone, `PhotoUploadWorker` will attempt to drain and upload Photographer A's photos under Photographer B's authentication token.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/ui/auth/AuthViewModel.kt`
    - `mobile/app/src/main/java/com/quickpitik/mobile/worker/PhotoUploadWorker.kt`
  - **Action Plan:** Clear or cancel pending upload queue items on logout, or enforce scoping by `photographerId` in `PhotoUploadWorker`.

- [ ] **20. Mobile — Download Photo Filename Collisions on Multi-Photo Downloads**
  - **Issue:** `PhotoDownloader.buildFilename(photoId, bib)` returns `quickpitik-bib-${bib.lowercase()}.jpg`. When a runner clicks "Download all" on an order containing multiple photos of the same bib number, every photo is assigned the identical filename `quickpitik-bib-101.jpg`. While MediaStore disambiguates with numerical suffixes, individual photo traceability is lost and filenames collide across events.
  - **Affected Files:**
    - `mobile/app/src/main/java/com/quickpitik/mobile/data/download/PhotoDownloader.kt`
  - **Action Plan:** Include the photo ID snippet in the filename: `quickpitik-bib-$bib-${photoId.take(8)}.jpg`.

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

---

### 📄 Documentation & Wireframes (SRS)

- [x] **15. SRS / Wireframe — Add updated Module 3 1.1 mobile wireframe**
  - **Status:** **ALREADY SOLVED**
  - **Verification:** Added `website/src/app/wireframes/srs/m3-1/page.tsx` rendering `UC-M3-1.1` Mobile Authentication & Verification wireframe. Registered route `UC-M3-1.1` in `website/src/app/wireframes/srs/page.tsx`.

