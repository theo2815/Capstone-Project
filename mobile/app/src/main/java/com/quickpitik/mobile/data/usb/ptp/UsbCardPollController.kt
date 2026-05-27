package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

/**
 * Zno-style USB card-poll capture for a PTP camera — the PRIMARY tether path.
 *
 * Instead of fighting the Canon EOS remote-capture extension (host RemoteRelease
 * needs Live View on the mirrorless R6, and a local shutter press isn't reported
 * to the host in remote mode), this reads the camera the boring, universal way:
 * enumerate the memory card over STANDARD PTP and pull each new shot. The
 * photographer just shoots normally — every frame lands on the card, we poll for
 * it, and pull the JPEG into the existing upload queue. This is the same approach
 * the commercial Zno Instant app uses on Canon/Nikon/Sony/Fuji over USB.
 *
 * Improvements over Zno's wired "Mark in Camera Mode" (which requires rating each
 * frame on the body): we upload EVERY new shot automatically, in capture order,
 * deduped by object handle, and pull only the JPEG (RAW stays on the card).
 *
 * The camera must be in its PC connection / PTP "Photo Import" USB mode with Auto
 * power off disabled. Takes a [deviceProvider] (not a fixed device) so each open
 * retry re-fetches the current handle if the body re-enumerated after sleeping.
 */
class UsbCardPollController(
    private val manager: UsbManager,
    private val deviceProvider: () -> UsbDevice?,
) {
    suspend fun run(
        onLog: (String) -> Unit,
        onCapture: suspend (filename: String, bytes: ByteArray) -> Unit,
    ) {
        val session = openWithRetry(onLog) ?: run {
            onLog(
                "Couldn't open the camera. Set its USB mode to PC connection / PTP " +
                    "(\"Photo Import\"), disable Auto power off, confirm the USB permission " +
                    "was allowed, then tap START CARD SYNC again."
            )
            return
        }
        try {
            // Baseline: every handle already on the card BEFORE we start. We only
            // upload shots taken after this point — and the set also dedupes, so a
            // reconnect never re-sends what we already pulled this session.
            val seen = HashSet<Long>()
            seen.addAll(runCatching { allHandles(session) }.getOrDefault(emptyList()))
            onLog(
                "Card sync live — ${seen.size} existing photo(s) ignored. " +
                    "Shoot away; each new frame uploads automatically."
            )

            var lastHeartbeat = System.currentTimeMillis()
            var consecutiveErrors = 0

            while (currentCoroutineContext().isActive) {
                val handles = try {
                    allHandles(session)
                } catch (e: Exception) {
                    consecutiveErrors++
                    if (consecutiveErrors == 1 || consecutiveErrors % 20 == 0) {
                        onLog("Card read error (${e.message}) ×$consecutiveErrors")
                    }
                    if (consecutiveErrors >= ERRORS_BEFORE_STOP) {
                        onLog("Camera keeps rejecting reads (${e.message}). Stopping — check the USB mode / power, then restart card sync.")
                        break
                    }
                    delay(POLL_MS)
                    continue
                }
                consecutiveErrors = 0

                // New handles, oldest first — Canon increments object handles, so
                // ascending order is capture order.
                val fresh = handles.filter { seen.add(it) }.sorted()
                for (handle in fresh) {
                    if (!currentCoroutineContext().isActive) break
                    val info = runCatching { session.getObjectInfo(handle) }.getOrNull()
                    if (info != null && (info.isAssociation || !info.isJpeg)) {
                        continue // folder or RAW — skip (RAW stays on the card)
                    }
                    val name = info?.filename?.takeIf { it.isNotBlank() }
                        ?: "IMG_%08X.jpg".format(handle)
                    val bytes = try {
                        session.getObject(handle)
                    } catch (e: Exception) {
                        onLog("  $name download failed: ${e.message}")
                        continue
                    }
                    if (bytes.isEmpty()) {
                        onLog("  $name returned 0 bytes — skipped")
                        continue
                    }
                    onLog("  $name ${bytes.size / 1024} KB → queued")
                    onCapture(name, bytes)
                }

                val now = System.currentTimeMillis()
                if (now - lastHeartbeat > HEARTBEAT_MS) {
                    onLog("Watching the card… shoot to upload.")
                    lastHeartbeat = now
                }
                delay(POLL_MS)
            }
        } finally {
            runCatching { session.close() }
            onLog("Card sync stopped.")
        }
    }

    /** Union of object handles across every storage; wildcard fallback if empty. */
    private fun allHandles(session: PtpSession): List<Long> {
        val all = LinkedHashSet<Long>()
        for (sid in runCatching { session.getStorageIds() }.getOrDefault(emptyList())) {
            runCatching { session.getObjectHandles(sid) }.getOrNull()?.let(all::addAll)
        }
        if (all.isEmpty()) {
            // Some bodies prefer the "all stores" wildcard (0xFFFFFFFF).
            runCatching { session.getObjectHandles(0xFFFFFFFFL) }.getOrNull()?.let(all::addAll)
        }
        return all.toList()
    }

    private suspend fun openWithRetry(onLog: (String) -> Unit): PtpSession? {
        for (attempt in 1..INIT_ATTEMPTS) {
            val device = deviceProvider()
            if (device == null) {
                onLog("No camera / USB permission (attempt $attempt)…")
                delay(RETRY_MS)
                continue
            }
            onLog(if (attempt == 1) "Opening camera…" else "Retry $attempt — opening camera…")
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
                // R6 treats a reused id 0 as a retransmit. So no GetDeviceInfo first.
                val rc = s.openSession()
                onLog("OpenSession rc=0x%04X".format(rc))
                if (rc != Ptp.RC_OK && rc != Ptp.RC_SESSION_ALREADY_OPEN) {
                    throw PtpException("OpenSession 0x%04X".format(rc))
                }
                return s
            } catch (e: Exception) {
                onLog("  open attempt $attempt failed: ${e.message}")
                runCatching { s.close() }
                delay(RETRY_MS)
            }
        }
        return null
    }

    private companion object {
        const val POLL_MS = 1200L
        const val HEARTBEAT_MS = 12000L
        const val INIT_ATTEMPTS = 4
        const val RETRY_MS = 1000L
        const val SETTLE_MS = 200L
        const val ERRORS_BEFORE_STOP = 12
    }
}
