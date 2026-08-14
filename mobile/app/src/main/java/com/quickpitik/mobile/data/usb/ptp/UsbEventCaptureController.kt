package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

/**
 * USB physical-shutter capture monitor for a Canon EOS body — live auto-upload:
 * the photographer SHOOTS NORMALLY over USB and each frame is pulled and handed
 * to [run]'s `onCapture` for the upload queue.
 *
 * The lever (from on-device findings): a *plain* PTP session ("PC connection")
 * locks the body's shutter, but **EOS remote + event mode keeps the body live —
 * the physical shutter fires and saves to the card** (journal 2026-05-26 §12).
 * Because Canon doesn't reliably report a locally-triggered capture through
 * `EOS_GetEvent` on every body/firmware, new shots are watched for TWO ways,
 * logging whichever the camera actually surfaces:
 *   1. `EOS_GetEvent` — an ObjectAddedEx for the freshly written frame.
 *   2. a throttled standard `GetObjectHandles` diff — in case the body doesn't
 *      report local captures but the card listing still grows.
 *
 * New frames are deduped by handle across both detectors; only JPEGs are pulled
 * (RAW stays on the card — filtered via `GetObjectInfo`, or an SOI byte sniff
 * when the camera won't give object info). A frame whose download fails is
 * retried the next time a detector surfaces it, up to [MAX_PULL_ATTEMPTS].
 *
 * Threading contract: every transfer is a synchronous `bulkTransfer` — callers
 * MUST launch [run] on `Dispatchers.IO`. Cancellation is only observed at
 * `delay()` points; an in-flight bulkTransfer finishes or times out (≤5 s)
 * first, so stopping can lag by up to that much. The `finally` block (event/
 * remote mode reset + session close) is entirely non-suspending and therefore
 * safe under cancellation.
 *
 * Takes a [deviceProvider] so each open retry re-fetches the current handle if the
 * camera re-enumerated after sleeping.
 */
class UsbEventCaptureController(
    private val manager: UsbManager,
    private val deviceProvider: () -> UsbDevice?,
) {
    private enum class PullOutcome { UPLOADED, SKIPPED, FAILED }

    suspend fun run(
        onLog: (String) -> Unit,
        onStarted: () -> Unit,
        onCapture: suspend (filename: String, bytes: ByteArray) -> Boolean,
    ) {
        val session = openAndInit(onLog) ?: run {
            onLog(
                "Couldn't start auto-upload. Wake the camera (half-press the shutter), set " +
                    "Auto power off → Disable, confirm the USB permission was allowed, then " +
                    "tap Start auto-upload again."
            )
            return
        }
        try {
            // Baseline BOTH detectors so existing frames aren't treated as new.
            val seen = HashSet<Long>()
            runCatching { EosEvents.parse(session.eosGetEvent()) }.getOrDefault(emptyList())
                .filter { it.isObjectArrival }.forEach { seen.add(it.handle) }

            // The card baseline is fail-closed: without it, the first card scan
            // would see EVERY photo on the card as new and upload all of them.
            var baseCard: List<Long>? = null
            for (attempt in 1..BASELINE_ATTEMPTS) {
                baseCard = runCatching { allCardHandles(session) }.getOrNull()
                if (baseCard != null) break
                onLog("Baseline attempt $attempt failed — retrying…")
                delay(RETRY_MS)
            }
            if (baseCard == null) {
                onLog(
                    "Couldn't read the card listing to set a baseline. Stopping so the " +
                        "whole card isn't re-uploaded — unplug/replug and try again."
                )
                return
            }
            seen.addAll(baseCard)
            onLog("Baseline set — ${baseCard.size} card object(s) known. Shoot now.")
            onStarted()
            onLog("Auto-upload live — press the shutter on the camera; each shot uploads.")

            // Download attempts per handle: retried whenever a detector surfaces
            // the handle again, then poisoned into `seen` at the cap so one bad
            // object can't stall the loop forever.
            val attempts = HashMap<Long, Int>()
            val loggedCodes = HashSet<Int>()
            var lastKeepAlive = System.currentTimeMillis()
            var lastCardScan = System.currentTimeMillis()
            var lastHeartbeat = System.currentTimeMillis()
            var consecutiveErrors = 0

            while (currentCoroutineContext().isActive) {
                // ── Detector 1: EOS event stream ────────────────────────────────
                val blob = try {
                    session.eosGetEvent()
                } catch (e: Exception) {
                    consecutiveErrors++
                    if (consecutiveErrors == 1 || consecutiveErrors % 25 == 0) {
                        onLog("GetEvent error (${e.message}) ×$consecutiveErrors")
                    }
                    if (consecutiveErrors >= ERRORS_BEFORE_STOP) {
                        onLog("Camera keeps rejecting GetEvent (${e.message}). Stopping — send me this log.")
                        break
                    }
                    delay(POLL_MS)
                    continue
                }
                consecutiveErrors = 0
                val events = EosEvents.parse(blob)
                for (ev in events) if (loggedCodes.add(ev.code)) onLog("event 0x%04X seen".format(ev.code))
                for (ev in events) {
                    if (!ev.isObjectArrival || ev.handle in seen) continue
                    settlePull(seen, attempts, ev.handle, pullViaEvent(session, ev.handle, onLog, onCapture), onLog)
                }

                val now = System.currentTimeMillis()

                // ── Detector 2: throttled card-listing diff ─────────────────────
                if (now - lastCardScan > CARD_SCAN_MS) {
                    lastCardScan = now
                    val handles = runCatching { allCardHandles(session) }.getOrNull()
                    if (handles != null) {
                        val fresh = handles.filter { it !in seen }.sorted()
                        if (fresh.isNotEmpty()) onLog("card-diff: ${fresh.size} new object(s) on card")
                        for (h in fresh) {
                            settlePull(seen, attempts, h, pullViaCard(session, h, onLog, onCapture), onLog)
                        }
                    }
                }

                if (now - lastHeartbeat > HEARTBEAT_MS) {
                    onLog("Watching… press the shutter to capture.")
                    lastHeartbeat = now
                }
                if (now - lastKeepAlive > KEEPALIVE_MS) {
                    runCatching { session.eosKeepDeviceOn() }
                    lastKeepAlive = now
                }
                delay(POLL_MS)
            }
        } finally {
            runCatching { session.eosSetEventMode(0) }
            runCatching { session.eosSetRemoteMode(0) }
            runCatching { session.close() }
            onLog("Auto-upload stopped.")
        }
    }

    /**
     * Fold a pull result into the dedupe/retry books. Success and deliberate
     * skips are final; failures earn another try the next time the handle
     * surfaces, up to [MAX_PULL_ATTEMPTS].
     */
    private fun settlePull(
        seen: HashSet<Long>,
        attempts: HashMap<Long, Int>,
        handle: Long,
        outcome: PullOutcome,
        onLog: (String) -> Unit,
    ) {
        when (outcome) {
            PullOutcome.UPLOADED, PullOutcome.SKIPPED -> {
                seen.add(handle)
                attempts.remove(handle)
            }
            PullOutcome.FAILED -> {
                val n = (attempts[handle] ?: 0) + 1
                attempts[handle] = n
                if (n >= MAX_PULL_ATTEMPTS) {
                    seen.add(handle)
                    onLog("handle 0x%08X gave up after $n attempts".format(handle))
                }
            }
        }
    }

    /** Pull a frame reported by the EOS event stream (EOS GetObject path). */
    private suspend fun pullViaEvent(
        session: PtpSession,
        handle: Long,
        onLog: (String) -> Unit,
        onCapture: suspend (String, ByteArray) -> Boolean,
    ): PullOutcome {
        // Standard GetObjectInfo works on event-surfaced handles (same handle
        // space the card path enumerates) and provides the type filter + the
        // real filename. In RAW+JPEG the camera reports BOTH objects — the RAW
        // must stay on the card.
        val info = runCatching { session.getObjectInfo(handle) }.getOrNull()
        if (info != null && info.isAssociation) return PullOutcome.SKIPPED
        if (info != null && !info.isJpeg) {
            onLog("non-JPEG (RAW) left on card — ${info.filename.ifBlank { "0x%08X".format(handle) }}")
            return PullOutcome.SKIPPED
        }
        val name = info?.filename?.takeIf { it.isNotBlank() } ?: "R6_%08X.jpg".format(handle)
        onLog("Capture via EVENT — $name (0x%08X)".format(handle))
        val bytes = try {
            session.eosGetObject(handle)
        } catch (e: Exception) {
            onLog("  EOS download failed: ${e.message}")
            return PullOutcome.FAILED
        }
        // 0 bytes may be an object still mid-write to the card — worth a retry.
        if (bytes.isEmpty()) { onLog("  0 bytes — will retry"); return PullOutcome.FAILED }
        runCatching { session.eosTransferComplete(handle) }
        if (info == null && !looksLikeJpeg(bytes)) {
            // No ObjectInfo to filter on — sniff the pulled bytes. A CR3 never
            // becomes a JPEG, so this is a skip, not a retry.
            onLog("  not a JPEG (no SOI) — left on card")
            return PullOutcome.SKIPPED
        }
        onLog("  $name ${bytes.size / 1024} KB → queued")
        return if (onCapture(name, bytes)) PullOutcome.UPLOADED else PullOutcome.FAILED
    }

    /** Pull a frame found by the standard card-listing diff (standard GetObject). */
    private suspend fun pullViaCard(
        session: PtpSession,
        handle: Long,
        onLog: (String) -> Unit,
        onCapture: suspend (String, ByteArray) -> Boolean,
    ): PullOutcome {
        val info = runCatching { session.getObjectInfo(handle) }.getOrNull()
        if (info != null && info.isAssociation) return PullOutcome.SKIPPED
        if (info != null && !info.isJpeg) return PullOutcome.SKIPPED // RAW stays on the card
        val name = info?.filename?.takeIf { it.isNotBlank() } ?: "IMG_%08X.jpg".format(handle)
        onLog("Capture via CARD — $name (0x%08X)".format(handle))
        val bytes = try {
            session.getObject(handle)
        } catch (e: Exception) {
            onLog("  card download failed: ${e.message}")
            return PullOutcome.FAILED
        }
        if (bytes.isEmpty()) { onLog("  0 bytes — will retry"); return PullOutcome.FAILED }
        if (info == null && !looksLikeJpeg(bytes)) {
            onLog("  not a JPEG (no SOI) — left on card")
            return PullOutcome.SKIPPED
        }
        onLog("  $name ${bytes.size / 1024} KB → queued")
        return if (onCapture(name, bytes)) PullOutcome.UPLOADED else PullOutcome.FAILED
    }

    /** JPEG SOI marker sniff — the filter of last resort when GetObjectInfo failed. */
    private fun looksLikeJpeg(bytes: ByteArray): Boolean =
        bytes.size >= 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte()

    /**
     * Union of object handles across every storage; wildcard fallback if empty.
     * Transport failures PROPAGATE — the baseline needs to distinguish "card is
     * empty" from "couldn't read the card" (a swallowed failure here is how an
     * empty baseline ends up re-uploading the entire card). The scan-loop call
     * site wraps with runCatching and just skips that tick.
     */
    private fun allCardHandles(session: PtpSession): List<Long> {
        val all = LinkedHashSet<Long>()
        for (sid in session.getStorageIds()) {
            all.addAll(session.getObjectHandles(sid))
        }
        if (all.isEmpty()) {
            all.addAll(session.getObjectHandles(0xFFFFFFFFL))
        }
        return all.toList()
    }

    /** Open + enter EOS event mode (the shutter stays live); retry a dozing body. */
    private suspend fun openAndInit(onLog: (String) -> Unit): PtpSession? {
        for (attempt in 1..INIT_ATTEMPTS) {
            val device = deviceProvider()
            if (device == null) {
                onLog("No camera / USB permission (attempt $attempt)…")
                delay(RETRY_MS)
                continue
            }
            onLog(if (attempt == 1) "Opening PTP session…" else "Retry $attempt — opening PTP session…")
            val s = try {
                PtpSession(manager, device)
            } catch (e: Exception) {
                onLog("  open failed: ${e.message}")
                delay(RETRY_MS)
                continue
            }
            try {
                delay(SETTLE_MS)
                // OpenSession must be the FIRST transaction (TransactionID 0) — the
                // R6 treats a reused id 0 as a retransmit and never really opens.
                val rc = s.openSession()
                onLog("OpenSession rc=0x%04X".format(rc))
                if (rc != Ptp.RC_OK && rc != Ptp.RC_SESSION_ALREADY_OPEN) {
                    throw PtpException("OpenSession 0x%04X".format(rc))
                }
                // EOS remote + event mode: this is the mode where the PHYSICAL
                // shutter still fires + saves to the card (journal §12). We do NOT
                // send RemoteRelease/CaptureDestination — those pushed the R6 into
                // Device_Busy. We only enter the mode, then READ new frames.
                s.eosSetRemoteMode(1)
                s.eosSetEventMode(1)
                onLog("EOS event mode on — shutter should stay live. Watching for shots…")
                return s
            } catch (e: Exception) {
                onLog("  init attempt $attempt failed: ${e.message}")
                runCatching { s.close() }
                delay(RETRY_MS)
            }
        }
        return null
    }

    private companion object {
        // Known latency characteristic (tuning hooks for the on-device session):
        // on an EMPTY event queue PtpSession.eosGetEvent() blocks up to ~3 s
        // (two 1.5 s readContainerOrNull legs around a clearHalt), so the
        // effective event-poll cadence is ~3.3 s, worst-case capture latency
        // ≈ 3–6.5 s including the card-scan fallback. Do NOT change PtpSession
        // read semantics without a camera in hand — a shorter read timeout
        // risks clearing a halt mid-data-phase and desyncing container framing.
        // Tune these constants using the timing log from the R6 verification
        // protocol, never blind.
        const val POLL_MS = 300L
        const val CARD_SCAN_MS = 3000L
        const val KEEPALIVE_MS = 4000L
        const val HEARTBEAT_MS = 10000L
        const val INIT_ATTEMPTS = 4
        const val RETRY_MS = 1000L
        const val SETTLE_MS = 200L
        const val ERRORS_BEFORE_STOP = 12
        const val BASELINE_ATTEMPTS = 3
        const val MAX_PULL_ATTEMPTS = 3
    }
}
