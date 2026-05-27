package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

/**
 * USB physical-shutter capture monitor for a Canon EOS body — the approach meant
 * to let the photographer SHOOT NORMALLY over USB and have each frame auto-upload.
 *
 * The lever (from on-device findings): a *plain* PTP session ("PC connection")
 * locks the body's shutter, but **EOS remote + event mode keeps the body live —
 * the physical shutter fires and saves to the card** (journal 2026-05-26 §12).
 * What was never solved is *reading the new frame* in that mode: Canon doesn't
 * report a locally-triggered capture through `EOS_GetEvent`. So this controller
 * enters the shutter-alive mode and watches for each new shot TWO ways, logging
 * whichever the R6 actually surfaces:
 *   1. `EOS_GetEvent` — an ObjectAddedEx for the freshly written frame.
 *   2. a throttled standard `GetObjectHandles` diff — in case the body doesn't
 *      report local captures but the card listing still grows.
 *
 * Every distinct event code + each detection (and whether the card listing grows)
 * is logged, so a single shooting session reveals exactly what works. New frames
 * are deduped by handle across both detectors and pulled with GetObject into the
 * existing upload queue.
 *
 * Takes a [deviceProvider] so each open retry re-fetches the current handle if the
 * R6 re-enumerated after sleeping.
 */
class UsbEventCaptureController(
    private val manager: UsbManager,
    private val deviceProvider: () -> UsbDevice?,
) {
    suspend fun run(
        onLog: (String) -> Unit,
        onCapture: suspend (filename: String, bytes: ByteArray) -> Unit,
    ) {
        val session = openAndInit(onLog) ?: run {
            onLog(
                "Couldn't start shutter watch. Wake the camera (half-press the shutter), set " +
                    "Auto power off → Disable, confirm the USB permission was allowed, then tap " +
                    "START SHUTTER WATCH again."
            )
            return
        }
        try {
            onLog("Shutter watch live — press the shutter on the camera; each shot should upload.")
            // Baseline BOTH detectors so existing frames aren't treated as new.
            val seen = HashSet<Long>()
            runCatching { EosEvents.parse(session.eosGetEvent()) }.getOrDefault(emptyList())
                .filter { it.isObjectArrival }.forEach { seen.add(it.handle) }
            val baseCard = runCatching { allCardHandles(session) }.getOrDefault(emptyList())
            seen.addAll(baseCard)
            onLog("Baseline set — ${baseCard.size} card object(s) known. Shoot now.")

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
                    if (!ev.isObjectArrival || !seen.add(ev.handle)) continue
                    pullViaEvent(session, ev.handle, onLog, onCapture)
                }

                val now = System.currentTimeMillis()

                // ── Detector 2: throttled card-listing diff ─────────────────────
                if (now - lastCardScan > CARD_SCAN_MS) {
                    lastCardScan = now
                    val handles = runCatching { allCardHandles(session) }.getOrNull()
                    if (handles != null) {
                        val fresh = handles.filter { seen.add(it) }.sorted()
                        if (fresh.isNotEmpty()) onLog("card-diff: ${fresh.size} new object(s) on card")
                        for (h in fresh) pullViaCard(session, h, onLog, onCapture)
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
            onLog("Shutter watch stopped.")
        }
    }

    /** Pull a frame reported by the EOS event stream (EOS GetObject path). */
    private suspend fun pullViaEvent(
        session: PtpSession,
        handle: Long,
        onLog: (String) -> Unit,
        onCapture: suspend (String, ByteArray) -> Unit,
    ) {
        onLog("Capture via EVENT — handle 0x%08X".format(handle))
        val bytes = try {
            session.eosGetObject(handle)
        } catch (e: Exception) {
            onLog("  EOS download failed: ${e.message}")
            return
        }
        if (bytes.isEmpty()) { onLog("  0 bytes — skipped"); return }
        runCatching { session.eosTransferComplete(handle) }
        onLog("  R6_%08X.jpg ${bytes.size / 1024} KB → queued".format(handle))
        onCapture("R6_%08X.jpg".format(handle), bytes)
    }

    /** Pull a frame found by the standard card-listing diff (standard GetObject). */
    private suspend fun pullViaCard(
        session: PtpSession,
        handle: Long,
        onLog: (String) -> Unit,
        onCapture: suspend (String, ByteArray) -> Unit,
    ) {
        val info = runCatching { session.getObjectInfo(handle) }.getOrNull()
        if (info != null && info.isAssociation) return
        if (info != null && !info.isJpeg) return // RAW stays on the card
        val name = info?.filename?.takeIf { it.isNotBlank() } ?: "IMG_%08X.jpg".format(handle)
        onLog("Capture via CARD — $name (0x%08X)".format(handle))
        val bytes = try {
            session.getObject(handle)
        } catch (e: Exception) {
            onLog("  card download failed: ${e.message}")
            return
        }
        if (bytes.isEmpty()) { onLog("  0 bytes — skipped"); return }
        onLog("  $name ${bytes.size / 1024} KB → queued")
        onCapture(name, bytes)
    }

    /** Union of object handles across every storage; wildcard fallback if empty. */
    private fun allCardHandles(session: PtpSession): List<Long> {
        val all = LinkedHashSet<Long>()
        for (sid in runCatching { session.getStorageIds() }.getOrDefault(emptyList())) {
            runCatching { session.getObjectHandles(sid) }.getOrNull()?.let(all::addAll)
        }
        if (all.isEmpty()) {
            runCatching { session.getObjectHandles(0xFFFFFFFFL) }.getOrNull()?.let(all::addAll)
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
        const val POLL_MS = 300L
        const val CARD_SCAN_MS = 3000L
        const val KEEPALIVE_MS = 4000L
        const val HEARTBEAT_MS = 10000L
        const val INIT_ATTEMPTS = 4
        const val RETRY_MS = 1000L
        const val SETTLE_MS = 200L
        const val ERRORS_BEFORE_STOP = 12
    }
}
