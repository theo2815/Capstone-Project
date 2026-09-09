# USB tethering implementation plan

Scope: Android USB ingestion into the existing Room/WorkManager/backend pipeline. Preserve physical-shutter Canon capture and on-card originals. Wi-Fi, remote shutter/live view, multi-camera operation, and RAW cloud processing are outside this increment. This repository has no iOS app; this work adds no iPhone USB implementation.

## Continuous auto-upload reliability — 2026-09-09

This increment preserves the September 7 drivers and durable spool. The Second
Brain's R6 physical-shutter/card-poll evidence and R10-only warm-up supersede older
notes that called R-series USB shooting impossible. USB is the intended transport:
leaving USB entirely would disconnect it. The fix is to retain supported live
camera control and stop with guidance if Canon rejects that setup; generic import
mode is no longer a silent fallback during a live shoot.

Connection recovery backs off from 2 to 30 seconds, with a five-minute recovery
budget that only a session healthy for ten seconds resets. Detached cameras wait
on USB connection state, with bounded timeout checks. A successful R10 attachment
is warmed once rather than before every reopen. Existing permission generation,
single-session ownership, camera identity and owner/event snapshots remain in force.

Card scans remain every three seconds because historical R6 EOS events missed
shots. New/event-signaled handles get metadata immediately; idle metadata checks
refresh at most eight existing handles every thirty seconds. Only changed sources
are reconciled into Room, and downloads reread identity before transferring a
delayed handle. Failed card transfers back off and stop after five attempts until
Retry; skipped failures cannot hide subsequent photos. Unplugged-camera and failed
source intent remains durable. Partial byte-range resume is not supported: an
interrupted camera file is safely retrieved again from its card.

Silent handle reuse without a detach or object event can take a full metadata
rotation to discover; this interval trades idle camera traffic for detection time.
Reconnecting or opening card import forces a fresh enumeration. Card-swap behavior
and sustained shooting throughput are hardware acceptance gates.

Sony's idle buffer checks back off from 0.5 to 3 seconds and drain available frames
sequentially. Its temporary fixed handle remains deliberately non-replayable after
an ambiguous transfer; recovery uses the required PC+Camera card backup. These
limits prevent attributing a later RAM frame to an earlier photo.

Room migration 2 → 3 adds the direct-upload checkpoint and retry deadline without
discarding queued originals. WorkManager coalesces bursts into one pending drain,
plus one successor when a drain is running, preserving the final-enqueue race.
The worker drains FIFO groups with bounded adaptive concurrency, stops a run after
an outage, and leaves long queues for subsequent runs. Completion order may differ
within a group; source identity and content hashes provide deduplication. The
[backend direct-upload contract](../../docs/BACKEND_ARCHITECTURE_ANALYSIS.md#73-photo-ingestion--watermarking-pipeline)
owns commit idempotency, object validation, retry classification and marketplace
publication details.

Signed PUT calls are cancellable, share the existing connection pool, and carry no
API authentication interceptor or signed-URL logging. Hashes computed during
camera transfer are reused. Transport failures retain originals without consuming
the five-attempt semantic failure budget; retries are paced by WorkManager rather
than an immediate loop. Refresh-service outages preserve login. The existing
Capture Retry action resets exhausted work after the cause is corrected.

Power changes remove upload-concurrency polling, repeated baseline database writes,
per-frame WorkManager chains, and unchanged notification updates. The foreground
service releases its wake lock while detached/inactive. Active Canon watching still
uses its established keep-alive; Stop releases live mode and the USB session. Do
not infer measured battery savings or guaranteed camera sleep while actively
watching. Android may defer networking/jobs under [Doze and App Standby](https://developer.android.com/training/monitoring-device-state/doze-standby),
and OEM process limits still require device testing. Durable capture/upload state
recovers on the next permitted run; force-stop is not a promise of continuous capture.

Approved owner-event selection now follows the backend's Manila calendar-month
window, including pre-event uploads. Pending/rejected events stay unavailable;
admin events keep their existing race-day/grace gate. No new cloud service,
dependency, camera reset, RAW upload path or AI call was added.

### Validation — reliability increment (2026-09-09)

From `mobile/`, `.\gradlew.bat --offline testDebugUnitTest assembleDebug lintDebug
compileReleaseKotlin --console=plain` passed: **134 tests, zero failures/errors/skips**,
debug APK assembly and release Kotlin compilation. Lint: **zero errors, 96 warnings,
7 informational findings**; dependency/deprecation and incompatible Compose custom
lint registry findings remain. Reports: `app/build/reports/tests/testDebugUnitTest/`
and `app/build/reports/lint-results-debug.html`. APK:
`app/build/outputs/apk/debug/app-debug.apk`.

From `backend/`, `.\gradlew.bat --offline test integrationTest --console=plain`
passed: **597 unit tests and 67 PostgreSQL integration tests**, zero
failures/errors/skips. Integration tests use local storage and PostgreSQL 16 in
Testcontainers, with AI disabled. Reports: `backend/build/reports/tests/`.

New regression evidence includes:

- A simulated 100-shot burst transfers all 100 post-baseline originals; reconnect
  catches up missed sources without retransferring completed ones. Reused handles
  are reidentified before transfer. Repeated failed sessions exhaust recovery;
  exhausted sources require Retry. One minute with 100 idle objects reads 108
  metadata records, rather than revisiting a slice every three-second scan.
- One hundred scheduling requests create one pending WorkManager drain. With a
  running drain they create exactly one successor. Cancellation closes a stalled
  PUT promptly, retains the original/checkpoint and later commits without another
  PUT. Lost PUT/commit acknowledgements, Room reopen, storage 503, HTTP 429 and
  refresh outages recover without redundant multipart delivery or lost retry intent.
- Four concurrent commits to one slot, and four same-content separate slots,
  return one photo, increment counters once and retain its original. The real
  asynchronous watermark worker publishes it LIVE. A later acknowledgement retry
  after the upload window closes still returns that photo. Wrong-owner IDs and
  mismatched/empty/oversized objects are rejected by focused service tests.

The initial backend regressions reproduced deletion of a referenced original and
an existing-ID overwrite before the fix. The full mobile run caught an uncaught
background scheduling exception after the async enqueue change; handling that
failure without losing queued intent made the complete suite pass. Existing
packet, migration, large-queue, format/size, ownership and lifecycle tests also pass.

ADB reports no attached devices. No live R2 bucket upload, physical shutter test,
real camera sleep/wake or power-saving run, independent card checksum comparison,
throughput measurement or battery A/B test was performed. Software coverage is
not production camera certification; complete the physical acceptance matrix below
on the target camera/phone before an unattended shoot.

## Compatibility expansion — 2026-09-07 investigation and implementation plan

This section supersedes the foundation's vendor-expansion deferral below. It was written before the expansion's code changes. Hardware observations in the vault are historical evidence, not certification of this revision.

### Findings and architecture decision

The active flow is `CameraConnectionManager -> foreground service/coordinator -> CameraDrivers -> PtpSession/PtpWire -> CameraTransferStore -> Room -> WorkManager`. Browse and import use the same driver/session factory. The unused EOS remote-shutter and card-poll controllers are historical experiments, not alternate production backends. Keep the service, durable spool, account/event boundaries and existing backend JPEG contract.

Use Android USB Host with standard PTP/MTP object access as the broad import baseline, capability-negotiated vendor adapters for live capture, and standard interrupt events plus periodic card reconciliation. Import, physical-shutter operation, transient host transfers and remote control are separate capabilities. A USB VID or a vendor-extension ID is insufficient proof of any of them. Do not require a model allowlist for standard PTP; use model quirks only for demonstrated exceptions.

The existing card-only assumption is a real limitation: Sony PC Remote delivers successive frames through the same temporary object handle. Those frames need their own ingestion path, bypassing the card baseline and durable handle suppression. Fetch each current object into the existing bounded, validated spool; preserve retry records but never replay a stale temporary handle against a different frame. Only allow this mode when the reported save destination includes both camera and host. Do not change exposure, trigger the shutter, switch capture destination or delete objects automatically. If the camera supplies RAW to the host, give JPEG-transfer setup guidance instead of deleting/consuming it speculatively.

### Alternatives evaluated

| Approach | Decision and evidence |
|---|---|
| Android `MtpDevice` | Native standard import and events, but its public API cannot send the Canon/Sony vendor commands required here. A second session owner would duplicate the current transfer path. Retain shared USB/PTP framing. [Android API](https://developer.android.com/reference/android/mtp/MtpDevice). |
| libgphoto2 + libusb | Best broad community-maintained protocol collection, including old proprietary cameras. Upstream supports Android's permitted FD through `gp_port_usb_set_sys_device`/`libusb_wrap_sys_device`; earlier claims that it only supports remote capture were incorrect. Adoption still needs NDK, autotools/cross-compilation, JNI, camera/port plugin packaging, FD lifecycle/cancellation and source/license distribution. This workspace has no NDK or general Linux build distribution (only Docker Desktop WSL); no camera is attached. Do not ship an unbuilt native wrapper. Implement the bounded vendor subset below, retaining the driver boundary for a separately built native backend. [Library](https://github.com/gphoto/libgphoto2), [FD implementation](https://github.com/gphoto/libgphoto2/blob/5672d510447bed26aa4c5d646665e768281bc680/libgphoto2_port/libusb1/libusb1.c), [libusb Android](https://github.com/libusb/libusb/blob/master/android/README). |
| Manufacturer SDKs | Reassessed independently of the vault. Fujifilm Camera Control SDK now explicitly supports Android USB; SDK distribution/EULA and its supported model/OS matrix must be assessed before incorporating binaries. Sony Camera Remote Command is published but download eligibility/terms apply; the cross-platform protocol facts are also implemented upstream. Canon EDSDK, Nikon SDK and Panasonic SDK availability does not establish broad Android support. [Fujifilm](https://www.fujifilm-x.com/global/camera-control-sdk/), [Sony](https://support.d-imaging.sony.co.jp/app/cameraremotecommand/en/index.html), [Panasonic](https://av.jpn.support.panasonic.com/support/global/cs/soft/tool/sdk.html). |
| Mass storage / UVC / Wi-Fi | USB mass storage needs filesystem access, not PTP; UVC is a video stream, not full-resolution originals; Wi-Fi changes the requested physical path. Report mode guidance rather than attempting PTP on these interfaces. |

Protocol research uses libgphoto2 commit `5672d510447bed26aa4c5d646665e768281bc680` (`camlibs/ptp2/{library.c,ptp.c,ptp.h,ptp-pack.c,config.c}`) as a reproducible reference, with independently implemented bounded Kotlin parsers. Reusing protocol facts is distinct from incorporating library code. Any future native distribution must preserve the component licenses/source obligations; the gphoto2 CLI is a separate GPL component and is not needed.

Standards/API references: [USB still-image transport definition and errata](https://www.usb.org/document-library/still-image-capture-device-definition-10-and-errata-16-mar-2007), [USB MTP 1.1 specification](https://www.usb.org/document-library/media-transfer-protocol-v11-spec-and-mtp-v11-adopters-agreement), [Android interrupt requests](https://developer.android.com/reference/android/hardware/usb/UsbRequest), [bounded requestWait and interface ownership](https://developer.android.com/reference/android/hardware/usb/UsbDeviceConnection). The existing minimum Android version covers these APIs; no minimum-version increase or new dependency was needed.

### Implementation sequence and verification

1. **Shared USB/PTP reliability.** One interface selector must require bulk IN/OUT and choose a viable interface/alternate consistently for discovery and transport. Read standard interrupt events through `UsbRequest` with bounded cancellation. Validate response parameter alignment, normalize truncated metadata errors, repair unknown-length header-only reads, and retry clean DeviceBusy only before data is delivered and for safe read/setup operations. Do not replay shutter triggers, partial files or transport failures. Test each failure with scripted USB packets and descriptor fixtures.
2. **Session and storage compatibility.** OpenSession first, then DeviceInfo, preserving the prior Canon R6 transaction-zero lesson and R10-only warm-up. Allow healthy slots to enumerate when another explicitly reports unavailable; propagate all other errors and do not turn an entirely failed scan into a baseline. Test command order, slot failures and fallback behavior.
3. **Vendor adapters.** Preserve Canon mode setup/card polling. Add Nikon advertised GetEvent/GetEventEx with bounded record parsers and a Nikon 1 exclusion for upstream's broken-event family. Add Sony SDIO connect stages 1/2, negotiated extended capabilities (request PTP 3; honor returned PTP 2), stage 3, and read-only property parsing for object availability/save destination. Gate Sony buffer ingest on supported operations and camera+host saving. Panasonic's card ObjectAdded event can wake standard reconciliation; host-memory-specific transfers remain explicitly unsupported. Test manufacturer/operation matching, negotiation, mixed property types and malformed arrays.
4. **Completeness and recovery.** Use events to prioritize accessible card objects while keeping periodic reconciliation for lost events. Add Sony's sequential temporary-object path outside numeric-handle dedupe, with fresh identity for every transfer attempt and content-hash dedupe only after bytes are durable. A failed temporary transfer stops with card-import guidance; it is not replayed automatically. Card backup is the recovery source. Test two different frames at the same handle, duplicate bytes, cancellation and absence of premature acknowledgement.
5. **Truthful setup/status.** Reuse the existing Capture UI and log tail to show manufacturer-specific USB/save mode guidance, distinguish USB attachment from camera readiness, and remove unconditional Canon shutter-lock language. No new design system or backend changes.
6. **Validate and document.** Run focused protocol/persistence tests, the full mobile unit suite, Kotlin compilation, debug assembly, Android lint and diff checks. Record build/tool failures separately from product failures. Update this document with final implementation details, deferred paths and exact validation results.

### Compatibility targets and limits

| Brand / generation | Import baseline | Live path in this expansion | Limits requiring evidence |
|---|---|---|---|
| Canon EOS DSLR/mirrorless | Standard PTP if advertised | Existing EOS mode plus card reconciliation; interrupt/vendor events where usable | R6/R10 history must be recertified; R10 warm-up stays narrow. Older PowerShot/EOS proprietary protocols are not automatically covered. Host-only EOS capture is not enabled. |
| Nikon DSLR/Z / older PTP | Standard PTP, healthy card slots | Standard events, advertised Nikon event polling, card reconciliation without PC-control mode writes | Nikon 1 vendor events can break sessions and are excluded. Storage-less SDRAM capture and shutter-unlocking behavior are not universally supported. |
| Sony Alpha / compatible SDIO generations | MTP/PTP in transfer mode | Negotiated SDIO PC Remote, property polling and sequential temporary JPEG objects with camera+host destination | Some older bodies lack these operations or simultaneous saving; QX/legacy DSLR protocols differ. PTP 2 remains a fallback, not the preferred protocol (Sony warns about future support). Body/firmware/mode testing required. |
| Fujifilm X/GFX / older FinePix | USB CARD READER mode when it exposes PTP/MTP (some modes/devices are mass storage) | Generic accessible card objects/events only | USB TETHER may be host-only and may not save to the card. Do not force undocumented PCMode/control transitions. Official Android SDK is a viable licensed extension, not integrated or claimed supported here. |
| Panasonic LUMIX | PTP/MTP mode when advertised | Standard/card-arrival events plus card reconciliation in a mode permitting shooting | PC(Storage) may be mass storage; PC(Tether) and models vary. Proprietary SDRAM/object acknowledgement flow and desktop SDK integration are not implemented. |

### Physical acceptance matrix

Test an older DSLR and newer mirrorless per available brand, rather than infer support from marketing names. Record exact body/firmware, USB VID:PID/interface/alternate, advertised operations, phone/Android version, USB/save format mode, cable and hub. Include two phone vendors. Start with R6 and R10 regression checks, then Nikon DSLR/Z and Sony generations returning protocol 2 and 3, then supported Fuji X/GFX and LUMIX bodies.

- Cold attach, permission denial/grant, replug, sleep/wake, stop/start and 20 repeated sessions; no competing USB owner or stale grant.
- Physical shutter, burst and sustained shoot, JPEG and RAW+JPEG (JPEG-to-host on Sony), two slots with one absent, card replacement and reused filenames/handles.
- Unplug during header/data/response, phone storage low, service cancellation/process death, offline upload/recovery and account change. Temporary frames must never be silently rebound to old metadata or a different event/account.
- Compare locally saved hashes with independently copied card originals before WorkManager shrinks completed files. Confirm both card originals and phone/cloud frames, not merely a successful USB response.
- Measure two-hour CPU/memory/battery and capture-to-phone latency. Use a data-capable cable/OTG host adapter suited to the camera port; validate powered USB-C PD hub and phone host+charging behavior. Software cannot fix charge-only cables, unsupported host mode, power instability or camera firmware limitations.

No physical camera is attached in this environment. Software tests can establish packet handling and persistence behavior; none of the five brand rows is a claim of universal or newly hardware-verified compatibility.

## Foundation implementation tasks (completed before the expansion)

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
| Protocol and model behavior | `data/usb/ptp/CameraDriver.kt`, `SonyPtpCamera.kt`, `PtpEvent.kt`: generic storage operations, Canon EOS setup/cleanup, Nikon event polling and negotiated Sony PC Remote. R10 warm-up only for `04a9:32f8`; rejected Canon live setup closes with actionable guidance. |
| Completeness and recovery | `data/usb/ptp/UsbEventCaptureController.kt`, `data/local/CameraTransferStore.kt`: durable baseline plus pending objects; reconcile all handles, periodically refresh existing metadata, retain gaps/failures. Manual selection is saved before transfer. Retry summary survives stopped watches. |
| Transfer and account boundary | `data/local/CameraTransferStore.kt`, `worker/PhotoUploadWorker.kt`, `data/remote/SessionAuthInterceptor.kt`: generated paths, bounded streaming, fsync/rename, local SHA-256, transactional enqueue, missing-file-safe dedupe, owner-specific draining and authentication refresh. |

Room migration 1 -> 2 adds the camera ledger; migration 2 -> 3 adds upload checkpoints/retry deadlines. Both preserve existing uploads; exported schemas are under `app/schemas/`. Existing UI and upload processing remain the integration path.

## Current compatibility boundary

Detection currently accepts USB still-image interfaces (class 6) with bulk IN/OUT; arbitrary vendor-specific interfaces and USB mass-storage/card-reader import require a subsequent adapter. Standard PTP card import is the first path for every brand that exposes the required operations. It is not a universal camera certification.

Card-based live ingest requires JPEG objects visible on camera storage while the physical shutter remains usable. Canon EOS enables remote/event mode without changing capture destination or deleting card originals; R6/R10 use card polling as established by prior hardware work. Nikon and standard interrupt events refresh card metadata; Panasonic card-arrival events are recognized. Sony SDIO has a separate temporary JPEG path gated on PC+Camera saving. Other vendor host-memory transfers are not implemented. Generic PTP modes that lock shooting require short manual card-import sessions.

The first card baseline intentionally excludes existing readable card photos; the UI reports monitoring afterward. Use manual import for older photos. Camera serial plus object metadata scopes recovery; without a serial, identity is attachment-scoped and replug may require a fresh start/manual import. Identical reused metadata cannot prove object continuity. Body/card changes never justify numeric high-water suppression. Sony's current buffer queue is drained without a card baseline, including frames already waiting when Start is pressed; changing events does not identify when those buffered frames were shot.

Temporary transfers use fresh attempt identities and content dedupe only after a complete download. Stop/cancel/disconnect may occur after the camera has advanced its buffer; retry never binds an old temporary handle to a new frame. Failed temporary records remain visible with card-import guidance and are excluded from automatic retries. Successfully renamed files are recoverable after restart. Manual card import cannot automatically prove which interrupted temporary record it replaces; that failure history is retained. Camera capture-mode settings, not simply the absence of DeleteObject, determine whether a card backup exists.

Guardrails: 16 KiB USB reads, bounded metadata/object payloads, 100,000 entries per PTP array, one camera at a time, full-file retry for stable card objects after interruption. Local length/signature/hash checks are not independently verified camera or cloud checksums. Stop/Pause retains pending work; Retry or a subsequent watch can resume selected card files, not temporary handles.

Sony setup tolerates response-only successful handshake stages and retries empty capability replies a bounded number of times. It requests protocol 3, accepts returned protocol 2, and retries with the older request only after a clean unsupported/invalid-parameter response. Pending-buffer and destination properties are refreshed before each transfer; no camera-setting writes are used. Unsupported property layouts fail explicitly. Newer secondary-enumeration parsing follows upstream's bounded `<512` count heuristic and needs fixtures from additional firmware generations before widening it. Interrupt requests stay queued between polls and are cancelled/closed with the session; their real detach/requeue timing remains a hardware gate.

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

## Further expansion

Evaluate a pinned libgphoto2/libusb Android backend before building more vendor stacks. Gate adoption on an unrooted-phone proof of permission/FD ownership, identification, exact file transfer, physical-shutter operation, card backup, cancellation/reconnection, ABI packaging and component licensing. Sony's published Camera Remote Command is a targeted alternative. Manufacturer support remains model/firmware/mode specific.

Follow-up order: (1) recertify R6/R10; (2) test the implemented Nikon events and Sony protocol 2/3 buffer paths; (3) build and validate a native backend for missing proprietary protocols, or a licensed Fujifilm Android SDK adapter; (4) expand the measured matrix to Fujifilm, Panasonic, Olympus/OM System, Leica and further models. Expose verified import/live capabilities per model rather than declaring whole brands supported. Buffer tuning, ledger retention and diagnostic export follow hardware measurements.

Manufacturer references correcting the historical vault: [Canon SDK platforms](https://www.usa.canon.com/support/sdk), [Nikon SDK updates](https://sdk.nikonimaging.com/information/en/), [Fujifilm Android SDK](https://www.fujifilm-x.com/global/camera-control-sdk/), [Fujifilm individual distribution agreement](https://www.fujifilm-x.com/global/camera-control-sdk/agreement/), [Sony command protocol and PTP 2 notice](https://support.d-imaging.sony.co.jp/app/cameraremotecommand/en/index.html), [Panasonic Windows SDK](https://av.jpn.support.panasonic.com/support/global/cs/soft/tool/sdk.html). No manufacturer agreement was accepted and no proprietary SDK was downloaded or bundled.

Camera setup references: [Sony PC+Camera destination](https://helpguide.sony.net/ilc/1640/v1/en/contents/TP0001306054.html), [Fujifilm X-T3 USB modes and card-saving warning](https://fujifilm-dsc.com/en/manual/x-t3/connections/computer/index.html), [Nikon MTP/PTP USB setting](https://onlinemanual.nikonimglib.com/z6III/en/nwm_usb_data_connection_365.html), [Panasonic USB mode settings](https://eww.pavc.panasonic.co.jp/dscoi/DC-S5M2X/html/DC-S5M2X_DVQP2992_eng/0125.html).

Primary references for that spike: [Android USB host](https://developer.android.com/develop/connectivity/usb/host), [libusb Android FD integration](https://github.com/libusb/libusb/blob/master/android/README), [libgphoto2 camera list and support distinctions](https://gphoto.sourceforge.io/proj/libgphoto2/support.php), [Sony Camera Remote Command](https://support.d-imaging.sony.co.jp/app/cameraremotecommand/en/index.html). Native integration is not a drop-in Gradle dependency.

## Future expansion and contract gaps

The USB reader carries generic object bytes; MVP publishes JPEG, including JPEG companions from RAW+JPEG. RAW archival/publishing requires backend content-type/size and processing support. Direct uploads currently accept JPEG/PNG/WebP up to 25 MiB and commit a client-asserted hash; independently verified cloud checksums are a separate backend change. Partial transfer resume requires verified camera support and stable object identity.

## Expansion file manifest

Paths below are relative to `mobile/`; the expansion changes 19 repository files. Earlier foundation changes are not counted again. No Gradle dependency, database schema, backend endpoint or website change was needed.

| Area | Changed files |
|---|---|
| Discovery | `app/src/main/java/com/quickpitik/mobile/data/usb/CameraConnectionManager.kt` |
| Shared protocol and drivers | `app/src/main/java/com/quickpitik/mobile/data/usb/ptp/`: `CameraDriver.kt`, `Ptp.kt`, `PtpDeviceInfo.kt`, `PtpSession.kt`, `PtpTransport.kt`, `PtpWire.kt`, `UsbEventCaptureController.kt`; new `PtpEvent.kt`, `SonyPtpCamera.kt` |
| Persistence | `app/src/main/java/com/quickpitik/mobile/data/local/`: `CameraTransfer.kt`, `CameraTransferStore.kt` |
| Existing Capture UI copy | `app/src/main/java/com/quickpitik/mobile/ui/photographer/DashboardScreen.kt` |
| Regression tests | `app/src/test/java/com/quickpitik/mobile/data/usb/ptp/`: `CameraDriverTest.kt`, `PtpWireTest.kt`; new `SonyPtpCameraTest.kt`, `UsbPtpInterfacesTest.kt`; `app/src/test/java/com/quickpitik/mobile/data/local/CameraTransferStoreTest.kt` |
| Plan/research/report | `docs/usb-tethering-plan.md` |

The vault's mobile task/decision notes, historical tethering guidance, dashboard status and a linked session journal were updated separately. Unrelated concurrent work in the shared workspace was preserved; no commit or deployment was performed by this session.

## Validation record — historical foundation

2026-09-07: `./gradlew.bat --offline testDebugUnitTest assembleDebug --console=plain` succeeded; **72 tests, 0 failures/errors/skips**. Includes framed/fragmented PTP, clean rejection vs invalid stream, cancellation, durable manual/live retry gaps, >200 catch-up objects, >1,000 queued uploads, Room v1 migration, crash-after-rename recovery, missing duplicate files and account/session isolation. Existing commerce tests also passed. `git diff --check` passed.

Debug APK: `app/build/outputs/apk/debug/app-debug.apk`. Existing SDK/Compose/Java-target deprecation warnings remain. The permission-state preview compiles; physical UI/permission timing, throughput and camera behavior are unverified in this increment. ADB reports no connected devices, and the configured Android SDK has no NDK installed. Native-backend adoption, hardware certification and RAW cloud processing remain open gates; no new camera support is claimed.

## Validation record — compatibility expansion (2026-09-07)

Final command from `mobile/`:

```powershell
.\gradlew.bat --offline testDebugUnitTest assembleDebug lintDebug compileReleaseKotlin --console=plain
```

**BUILD SUCCESSFUL.** The final JUnit XML reports **106 tests, 0 failures, 0 errors, 0 skipped**. This expansion adds 28 regressions across the five protocol/persistence test files; totals include other existing/concurrent mobile tests. Coverage includes session transaction order, multi-slot failures, composite/alternate USB descriptors, Nikon event formats/fallback, unsigned handles, Busy retry/cancellation, unknown-length header/ZLP boundaries, Sony protocol 2/3 and empty handshakes, mixed property datasets, unsafe save modes, repeated RAM handles, content dedupe, interrupted temporary transfers and recovery after a completed rename. These are synthetic packet/Room fixtures, not camera USB traces.

Debug and release Kotlin compilation passed; the debug APK is at `app/build/outputs/apk/debug/app-debug.apk`. Android lint completed with **0 errors, 95 warnings and 7 informational findings** (`app/build/reports/lint-results-debug.xml`/`.html`). Existing dependency/deprecation findings remain, including `StaticFieldLeak` on the unchanged application-context coordinator. The existing Compose custom-check registry is incompatible with this lint version (`ObsoleteLintCustomCheck`), so its listed checks were skipped; this is not a claim of complete Compose static-analysis coverage. No lint issue was reported on the new protocol/transfer implementation.

`git diff --check -- mobile` and the scoped vault diff checks passed. An independent read-only protocol/recovery review found no additional blocking issues. The first sandboxed Gradle attempt could not access its global cache; the approved cache-access runs above succeeded without changing dependency versions. Unrelated workspace changes were not reverted or included as tethering work.

ADB listed no attached devices. No phone installation, physical-shutter test, card-original checksum comparison, power/cable measurement or camera certification was performed. Follow the physical acceptance matrix before production shoots. Sony descriptor variations that omit fields, vendor-only interfaces, licensed Fuji/other proprietary modes and native-backend packaging remain explicit compatibility gates—not silently accepted layouts or promised support.
