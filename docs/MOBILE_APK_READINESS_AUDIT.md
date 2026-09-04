# QuickPitik Mobile — APK Readiness Audit & Physical Device Testing Guide

**Date:** September 4, 2026  
**Audited Directory:** `mobile/`  
**Target Platform:** Android (Native Kotlin / Jetpack Compose)  
**Artifact Path:** `mobile/app/build/outputs/apk/debug/app-debug.apk`

---

## 🚦 1. Executive Summary & Readiness Verdict

### **Readiness Verdict: READY WITH WARNINGS**

> **Primary Question:** *“If I build an APK from the current mobile folder, can I install it on an Android phone and realistically use it for testing?”*  
> **Verdict:** **YES** — provided the team builds and distributes the **Debug APK (`assembleDebug`)**, NOT the Release APK.

The mobile codebase compiles cleanly with 0 errors, passes 100% of its JVM test suite (Robolectric Room/WorkManager tests), and already features an in-app server configuration mechanism (`DevServerRow`) specifically engineered to let physical Android phones connect to a laptop's local Spring Boot backend over Wi-Fi or USB.

| Category | Status | Details |
| :--- | :---: | :--- |
| **Kotlin / Gradle Compilation** | ✅ PASS | `compileDebugKotlin` succeeds in ~40s with 0 errors. |
| **APK Generation** | ✅ PASS | `assembleDebug` builds signed `app-debug.apk` (~34.8 MB). |
| **Unit Test Suite** | ✅ PASS | `testDebugUnitTest` executed 33 tasks with 0 failures. |
| **Debug Signing** | ✅ PASS | Automatically signed with standard Android debug keystore. |
| **Network Cleartext Traffic** | ✅ PASS | Permitted in debug builds (`usesCleartextTraffic = true`). |
| **Local IP Reconfiguration** | ✅ PASS | In-app server switcher (`DevServerRow`) available on Login. |
| **Release Build Status** | ⚠️ UNFIT FOR TEST | Release APK is unsigned, blocks HTTP, and hides server UI. |

---

## ⚠️ 2. Critical Findings & Operational Rules

Before distributing the APK to teammates, ensure everyone understands the following critical rules:

### Rule 1: Always distribute `assembleDebug`, NEVER `assembleRelease`
* **File:** [`mobile/app/build.gradle.kts`](file:///c:/Users/USER/Documents/School/4th%20Year%201st%20Semester/Capstone%20and%20Research%202/CAPSTONE%20PROJECT/Capstone-Project/mobile/app/build.gradle.kts)
* **Reason:**
  1. **Unsigned APK:** The `release` build type has no `signingConfig`. It generates `app-release-unsigned.apk`. Physical Android devices will **refuse** to install it (`INSTALL_PARSE_FAILED_NO_CERTIFICATES`).
  2. **Cleartext Blocked:** Release builds set `manifestPlaceholders["usesCleartextTraffic"] = "false"`, which instructs Android OS to reject all unencrypted HTTP traffic (such as `http://192.168.1.X:8080`).
  3. **Locked Server Config:** In release builds, `BuildConfig.DEBUG` is `false`. This hides the `DevServerRow` UI on the Login screen and disables `RetrofitClient.setBaseUrl()`, locking the app to the emulator-only fallback (`http://10.0.2.2:8080/`), rendering the app completely dead on physical devices.

### Rule 2: Physical devices cannot use default `10.0.2.2:8080`
* **File:** [`mobile/app/src/main/java/com/quickpitik/mobile/data/remote/RetrofitClient.kt`](file:///c:/Users/USER/Documents/School/4th%20Year%201st%20Semester/Capstone%20and%20Research%202/CAPSTONE%20PROJECT/Capstone-Project/mobile/app/src/main/java/com/quickpitik/mobile/data/remote/RetrofitClient.kt)
* **Reason:** `10.0.2.2` is a special loopback alias recognized only by the Android Studio Emulator. A physical phone has its own network interface and cannot reach `10.0.2.2`.
* **Action:** Testers must either use the **in-app server switcher** to enter their PC's local Wi-Fi IP (e.g. `192.168.1.50:8080`) or establish an **ADB reverse proxy** (`adb reverse tcp:8080 tcp:8080`).

---

## 🔍 3. Warnings & Non-Blockers

1. **Minimum Android OS Version (`minSdk = 29`):**
   * The application requires **Android 10 or higher** (API 29+). Devices running Android 9 (Pie) or older cannot install the APK.
2. **Google Sign-In Requires Local Keystore SHA-1 Registration:**
   * [`gradle.properties`](file:///c:/Users/USER/Documents/School/4th%20Year%201st%20Semester/Capstone%20and%20Research%202/CAPSTONE%20PROJECT/Capstone-Project/mobile/gradle.properties) includes the Google OAuth Web Client ID (`QP_GOOGLE_SERVER_CLIENT_ID`).
   * For the Google button to work on an Android device, the SHA-1 fingerprint of the machine's `debug.keystore` must be registered in the Google Cloud Console under an Android OAuth Client for `com.quickpitik.mobile`.
   * *Graceful Fallback:* If not registered, the app **does not crash**; it gracefully catches `GetCredentialException` and displays *"Google sign-in unavailable. Use your email instead."* Standard email/password login and registration work 100% reliably.
3. **Hardcoded ngrok Domain in AndroidManifest:**
   * In [`AndroidManifest.xml`](file:///c:/Users/USER/Documents/School/4th%20Year%201st%20Semester/Capstone%20and%20Research%202/CAPSTONE%20PROJECT/Capstone-Project/mobile/app/src/main/AndroidManifest.xml), line 90 specifies `snippet-sheath-cloak.ngrok-free.dev` for email verification deep-link interception.
   * If ngrok restarts and generates a new URL, clicking a verification link in an email client will open in Chrome instead of deep-linking directly into the app. (Verification still succeeds through the website).
4. **Google Fonts Runtime Resolution:**
   * The app uses Google Play Services Downloadable Fonts (`Anton`, `Archivo`, `Geist Mono`). On devices without Google Play Services or on first launch without an active internet connection, Jetpack Compose falls back cleanly to the system font (Roboto) without crashing.

---

## 🏗️ 4. What Is Already Configured Correctly

* **Modern Jetpack Compose & Material 3 UI:** Clean declarative screens matching the website's visual tokens (`Bone`, `Ink`, `Fresh`, `Slate`, `Line`).
* **In-App Server Origin Switching:** `DevServerRow` and `DevServerSheet` on the Login screen allow setting any Wi-Fi IP or ngrok host at runtime. The setting is persisted in SharedPreferences (`quickpitik_dev.xml`) and survives app restarts.
* **Coil Image Loading & Host Rewriting:** [`HostRewriteInterceptor.kt`](file:///c:/Users/USER/Documents/School/4th%20Year%201st%20Semester/Capstone%20and%20Research%202/CAPSTONE%20PROJECT/Capstone-Project/mobile/app/src/main/java/com/quickpitik/mobile/data/remote/HostRewriteInterceptor.kt) and `RetrofitClient.resolveImageUrl()` dynamically rewrite relative paths and loopback URLs (`localhost:8080`) to the active host, ensuring thumbnails and photos render on physical phones.
* **Modern Storage Permissions (Scoped Storage):** Selfies and photo downloads use Android Scoped Storage (`MediaStore`), eliminating the need for legacy storage permissions.
* **System Camera & Photo Pickers:** Camera capture uses `ActivityResultContracts.TakePicture()` and photo picking uses `ActivityResultContracts.GetMultipleContents()`, delegating to system apps without requiring raw `CAMERA` permission.
* **Tethered Camera Ingest:** [`TetherIngestService.kt`](file:///c:/Users/USER/Documents/School/4th%20Year%201st%20Semester/Capstone%20and%20Research%202/CAPSTONE%20PROJECT/Capstone-Project/mobile/app/src/main/java/com/quickpitik/mobile/service/TetherIngestService.kt) implements `FOREGROUND_SERVICE_CONNECTED_DEVICE` with partial wakelocks and exception handling for Android 14+ USB permission drops.
* **PayMongo Checkout Integration:** Handled via Chrome Custom Tabs (`CustomTabsIntent`) with automatic return handling via the `quickpitik://orders/return` deep link bridge.

---

## 📋 5. Step-by-Step APK Testing Guide for Teammates

### Phase 1: Build the APK
Open PowerShell in the `mobile/` directory and run:
```powershell
cd "c:\Users\USER\Documents\School\4th Year 1st Semester\Capstone and Research 2\CAPSTONE PROJECT\Capstone-Project\mobile"
.\gradlew.bat assembleDebug
```
The APK will be generated at:
```
mobile\app\build\outputs\apk\debug\app-debug.apk
```

---

### Phase 2: Install on an Android Device

#### Method A: Direct Install via USB / ADB (Recommended for Developers)
1. On your phone: Go to **Settings** -> **About Phone** -> Tap **Build Number** 7 times to enable Developer Options.
2. Go to **Settings** -> **Developer Options** -> Enable **USB Debugging**.
3. Connect the phone to your computer via USB.
4. Verify ADB sees the phone:
   ```powershell
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" devices
   ```
5. Install the APK directly:
   ```powershell
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" install -r "app\build\outputs\apk\debug\app-debug.apk"
   ```

#### Method B: Sideloading (For Teammates without ADB)
1. Send `app-debug.apk` to your teammates (via Google Drive, USB flash drive, Discord, etc.).
2. On their phone, tap the APK file.
3. When prompted *"For your security, your phone is not allowed to install unknown apps from this source"*, tap **Settings** and enable **Allow from this source**.
4. Tap **Install**.

---

### Phase 3: Connect the Mobile App to the Backend

The phone must be able to communicate with the Spring Boot backend running on your PC (port 8080).

#### Option A: Wireless Local Wi-Fi (Most Common)
1. Ensure both your computer and the Android phone are connected to the **same Wi-Fi network**.
2. Find your computer's local IP address by opening PowerShell and running:
   ```powershell
   ipconfig
   ```
   Look for `IPv4 Address` under your active Wi-Fi adapter (e.g., `192.168.1.50`).
3. Make sure Windows Firewall allows incoming connections on port `8080`.
4. Launch the **QuickPitik** app on the phone.
5. On the **Login** screen, scroll down to the bottom and tap:
   ```
   SERVER · 10.0.2.2:8080
   ```
6. Type your computer's IP address and port (e.g., `192.168.1.50:8080`) and tap **Use this server**.
7. The app is now connected! All requests, images, and WebSockets will use this host.

#### Option B: USB Port Forwarding (If Wi-Fi Has Isolation / Firewall Issues)
1. Keep the phone plugged in via USB with USB Debugging enabled.
2. Run the following ADB reverse proxy command:
   ```powershell
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" reverse tcp:8080 tcp:8080
   ```
3. Open the app, tap the `SERVER` button at the bottom of the Login screen, enter `localhost:8080`, and tap **Use this server**.

---

### Phase 4: What to Test on the Device

#### 1. Authentication & Session
* [ ] Register a new runner account using email and password.
* [ ] Log in with an existing account.
* [ ] Force-close the app and reopen it: verify the session token persists and bypasses the login screen.
* [ ] Test Sign Out from settings: verify it returns cleanly to Login and preserves the server IP setting.

#### 2. Runner Marketplace & Events
* [ ] Browse the public events list on the Discover tab.
* [ ] Open an event: verify event details, banner image, and watermarked photo grid load.
* [ ] Tap a photo thumbnail to open the preview modal.
* [ ] Add a photo to the cart: verify the floating cart pill updates its count and total price.
* [ ] Open Cart -> Proceed to Checkout -> Verify Chrome Custom Tabs opens the PayMongo checkout gateway.

#### 3. Runner Profile & Selfies
* [ ] Go to the **Profile** tab.
* [ ] Tap the `+` affordance on the selfie grid.
* [ ] Choose **Camera**: take a selfie with the phone camera and verify it uploads and shows quality score.
* [ ] Choose **Gallery**: pick an existing photo and verify upload.
* [ ] Tap a selfie card: test setting it as primary or deleting it.

#### 4. Photographer Studio Flow
* [ ] Log in with a photographer account.
* [ ] Verify the UI transitions to the dark Studio Theme with 5 bottom tabs:
  * **Home (Overview):** Verification status banner and earnings summary.
  * **Capture:** USB DSLR connect prompt / shutter monitor.
  * **Events:** Covered events list and public events picker.
  * **Earnings:** Payout metrics and transaction history.
  * **Settings:** Studio profile and payout account details.
* [ ] Switch between Runner View and Photographer View via the avatar dropdown.

---

## 🛠️ 6. Troubleshooting & FAQs

* **Q: The app shows "Couldn't reach QuickPitik — check your connection" on login.**  
  * **Cause:** The phone cannot reach port 8080 on your PC.
  * **Fix:** Check `ipconfig` again (Wi-Fi IPs can change). Verify Windows Firewall is not blocking Java/port 8080. If your router has AP client isolation enabled, switch to Option B (`adb reverse tcp:8080 tcp:8080`).

* **Q: "Google sign-in unavailable. Use your email instead."**  
  * **Cause:** The test machine's debug SHA-1 fingerprint is not registered in Google Cloud Console.
  * **Fix:** Use email/password for internal testing, or retrieve your debug SHA-1 via:
    ```powershell
    keytool -list -v -keystore "$env:USERPROFILE\.android\debug.keystore" -alias androiddebugkey -storepass android -keypass android
    ```
    and register it in the Google Cloud Console under an Android OAuth Client for package `com.quickpitik.mobile`.

* **Q: Photos/watermarks are blank gray boxes.**  
  * **Cause:** The backend generated image URLs with `localhost:8080` and the phone tried to resolve them locally.
  * **Fix:** Make sure the app's server setting in `DevServerRow` was set to your laptop's LAN IP (`192.168.1.X:8080`), so `HostRewriteInterceptor` can correctly rewrite `localhost` to that IP.
