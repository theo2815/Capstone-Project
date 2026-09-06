# USB tethering implementation plan

Scope: Android USB ingestion into the existing Room/WorkManager/backend pipeline. Preserve physical-shutter Canon capture and on-card originals. Wi-Fi, remote shutter/live view, multi-camera operation, and RAW cloud processing are outside this increment.

## Implementation tasks

- [x] Harden the shared PTP transport: validate framing, transaction IDs, short writes, lengths and metadata bounds; invalidate interrupted sessions; add scripted transport regression tests.
- [x] Separate generic PTP and Canon EOS behavior using detected operations and model profiles. Scope the R10 warm-up. Route browse/import/watch through exclusive USB ownership and close failed initialization.
- [x] Represent permission pending/denied separately from a usable connection; bind permission callbacks and sessions to the selected attachment.
- [x] Persist camera discovery and pending transfers in Room. Replace event-only high-water and filename-only suppression; preserve failures and baselines across reconnect/restart.
- [x] Write safe generated `.part` files, validate size/JPEG signature, sync/rename and transactionally enqueue; reconcile interrupted persistence and pause on low storage.
- [x] Isolate queue operations by account; make upload scheduling reliable across the last-enqueue race and authentication changes; clean files when deleting failed rows.
- [x] Give the foreground service ownership of ingestion lifetime; serialize cancellation and teardown; expose actionable recovery state through existing UI.
- [x] Run focused regression tests, the mobile unit suite, Kotlin compilation and debug APK assembly. Inspect the final diff and update this plan with actual results.

## Implemented flow and extension points

`CameraConnectionManager -> CameraIngestCoordinator / TetherIngestService -> CameraDrivers -> PtpSession / PtpWire -> CameraTransferStore -> Room upload queue -> PhotoUploadScheduler / PhotoUploadWorker`

Source paths below are relative to `app/src/main/java/com/quickpitik/mobile/`.

| Responsibility | Implementation |
|---|---|
| Attachment and permission | `data/usb/CameraConnectionManager.kt`: selects an attachment, checks actual permission, rejects stale callbacks; explicit denial/retry UI. |
| Ownership and lifetime | `data/usb/CameraIngestCoordinator.kt`, `service/TetherIngestService.kt`: service-owned jobs, matching start-command handshake, cancellation completion before reuse, one USB owner. Live capture survives dashboard navigation; account-bound event snapshot restores offline. |
| Protocol and model behavior | `data/usb/ptp/CameraDriver.kt`: generic storage operations plus Canon EOS setup/cleanup. Canon detection combines VID/manufacturer with advertised operations, not VendorExtensionID alone. R10 warm-up only for `04a9:32f8`; clean OperationNotSupported opens a fresh generic session. |
| Completeness and recovery | `data/usb/ptp/UsbEventCaptureController.kt`, `data/local/CameraTransferStore.kt`: durable baseline plus pending objects; reconcile all handles, periodically refresh existing metadata, retain gaps/failures. Manual selection is saved before transfer. Retry summary survives stopped watches. |
| Transfer and account boundary | `data/local/CameraTransferStore.kt`, `worker/PhotoUploadWorker.kt`, `data/remote/SessionAuthInterceptor.kt`: generated paths, bounded streaming, fsync/rename, local SHA-256, transactional enqueue, missing-file-safe dedupe, owner-specific draining and authentication refresh. |

Room migration 1 -> 2 preserves existing uploads and adds the camera ledger; exported schema is under `app/schemas/`. Existing UI, backend contracts and upload processing remain the integration path.

## Current compatibility boundary

Detection currently accepts USB still-image interfaces (class 6) with bulk IN/OUT; arbitrary vendor-specific interfaces and USB mass-storage/card-reader import require a subsequent adapter. Standard PTP card import is the first path for every brand that exposes the required operations. It is not a universal camera certification.

Live ingest currently requires JPEG objects visible on camera storage while the physical shutter remains usable. Canon EOS enables remote/event mode without changing capture destination or deleting card originals; R6/R10 use card polling as established by prior hardware work. Other EOS event notifications are reconciliation hints; host-memory-only event transfer is not implemented in this increment. Generic PTP modes that lock shooting fall back to short manual card-import sessions.

The first live baseline intentionally excludes existing card photos; the UI reports readiness afterward. Use manual import for older photos. Camera serial plus object metadata scopes recovery; without a serial, identity is attachment-scoped and replug may require a fresh start/manual import. Identical reused metadata cannot prove object continuity. Body/card changes never justify numeric high-water suppression.

Guardrails: 16 KiB USB reads, bounded metadata/object payloads, 100,000 entries per PTP array, one camera at a time, full-file retry after interruption. Local length/signature/hash checks are not independently verified camera or cloud checksums. Stop/Pause retains pending work; Retry or a subsequent watch can resume selected files.

## Verification gates

Software tests cover malformed/truncated PTP, unsupported commands, retry gaps, interrupted persistence, duplicate filenames/handles, account isolation, and cancellation/ownership. Keep the existing 16 KiB transfer buffer until device measurements justify changing it.

Hardware release gate (requires available bodies/phone): R6 and R10 plus at least one Sony/Nikon; repeated cold attach/permission/reconnect, physical-shutter bursts, JPEG and RAW+JPEG, card replacement, offline upload, low storage, long sessions, and comparison against independently copied camera originals. Compilation or synthetic tests do not certify cameras.

| Hardware check | Pass condition |
|---|---|
| Cold attach, deny, grant, stale callback, replug; repeat 20 times | No competing sessions or permission loops; clear recovery action after denial. |
| 100 JPEG shots and a 300-photo reconnect backlog | Every eligible source is queued or visibly pending/failed, no silent omissions, card originals retained. |
| RAW-only, RAW+JPEG, duplicate names, reused handles, card change | Clear format guidance; JPEG companions imported; no wrong-source publication. |
| Unplug/cancel during read; kill after rename; navigate and recreate dashboard | Partial files never upload; durable work recovers; running shoot keeps the same account/event. |
| Offline/low storage, then recovery; logout/login another account | No lost retry intent, no cross-account publication, uploads resume for the owning account. |
| Two-hour session on two phone models | Record p50/p95 capture-to-local and local-to-cloud latency, memory and battery, compare with prior build; no sustained queue growth at the tested shooting rate. |

Record exact body, firmware, camera USB/save mode, phone/Android version and cable/hub for each result. Compare hashes against independently copied originals before the upload worker replaces completed local files with preview thumbnails.

## Important improvements / investigation

Evaluate a pinned libgphoto2/libusb Android backend before building more vendor stacks. Gate adoption on an unrooted-phone proof of permission/FD ownership, identification, exact file transfer, physical-shutter operation, card backup, cancellation/reconnection, ABI packaging and component licensing. Sony's published Camera Remote Command is a targeted alternative. Manufacturer support remains model/firmware/mode specific.

Follow-up order: (1) recertify R6/R10; (2) probe Sony/Nikon generic import and shooting modes, then the native-backend spike; (3) add only the demonstrated missing vendor operations/model quirks; (4) expand the measured matrix to Fujifilm, Panasonic, Olympus/OM System, Leica and further models. Expose verified import/live capabilities per model rather than declaring whole brands supported. Buffer tuning, interrupt-endpoint events, ledger retention and diagnostic export follow hardware measurements.

Primary references for that spike: [Android USB host](https://developer.android.com/develop/connectivity/usb/host), [libusb Android FD integration](https://github.com/libusb/libusb/blob/master/android/README), [libgphoto2 camera list and support distinctions](https://gphoto.sourceforge.io/proj/libgphoto2/support.php), [Sony Camera Remote Command](https://support.d-imaging.sony.co.jp/app/cameraremotecommand/en/index.html). Native integration is not a drop-in Gradle dependency.

## Future expansion and contract gaps

The USB reader carries generic object bytes; MVP publishes JPEG, including JPEG companions from RAW+JPEG. RAW archival/publishing requires backend content-type/size and processing support. Direct uploads currently accept JPEG/PNG/WebP up to 25 MiB and commit a client-asserted hash; independently verified cloud checksums are a separate backend change. Partial transfer resume requires verified camera support and stable object identity.

## Validation record

2026-09-07: `./gradlew.bat --offline testDebugUnitTest assembleDebug --console=plain` succeeded; **72 tests, 0 failures/errors/skips**. Includes framed/fragmented PTP, clean rejection vs invalid stream, cancellation, durable manual/live retry gaps, >200 catch-up objects, >1,000 queued uploads, Room v1 migration, crash-after-rename recovery, missing duplicate files and account/session isolation. Existing commerce tests also passed. `git diff --check` passed.

Debug APK: `app/build/outputs/apk/debug/app-debug.apk`. Existing SDK/Compose/Java-target deprecation warnings remain. The permission-state preview compiles; physical UI/permission timing, throughput and camera behavior are unverified in this increment. ADB reports no connected devices, and the configured Android SDK has no NDK installed. Native-backend adoption, hardware certification and RAW cloud processing remain open gates; no new camera support is claimed.
