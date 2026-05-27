package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/** One Canon EOS event record pulled from an EOS_GetEvent payload. */
data class EosEvent(val code: Int, val handle: Long) {
    val isObjectArrival: Boolean
        get() = code == Ptp.EOS_EVENT_OBJECT_ADDED_EX ||
            code == Ptp.EOS_EVENT_OBJECT_ADDED_EX64 ||
            code == Ptp.EOS_EVENT_REQUEST_OBJECT_TRANSFER
}

/**
 * Drives real shutter → auto-upload for a Canon EOS body (verified: EOS R6).
 *
 * The body doesn't push captures on the USB interrupt endpoint, so after we put
 * it in remote + event mode we poll `EOS_GetEvent` and react to the events it
 * carries. On a shutter press the camera emits `ObjectAddedEx` (0xC181 / 0xC191)
 * — or, in PC-save mode, `RequestObjectTransfer` (0xC186) — naming a new object
 * handle. We pull the JPEG with `GetObject`, ack with `TransferComplete`, and
 * hand the bytes to [run]'s `onCapture`, which the ViewModel enqueues into the
 * existing upload pipeline.
 *
 * Init: `GetDeviceInfo` (wakes/primes the body) → `CloseSession` (clears any
 * session a prior probe left bound to a now-dead connection — the cause of
 * `GetEvent → Session_Not_Open`) → `OpenSession` → `SetRemoteMode(1)` →
 * `SetEventMode(1)`, retried on a fresh session because the R6 dozes between the
 * probe and the START tap. Response codes are logged so a real run is diagnosable.
 *
 * Takes a [deviceProvider] (not a fixed device) so each retry re-fetches the
 * current handle in case the camera re-enumerated after sleeping.
 */
class EosTetherController(
    private val manager: UsbManager,
    private val deviceProvider: () -> UsbDevice?,
) {
    // Set from the UI thread, consumed on the poll coroutine — so ALL session I/O
    // (GetEvent + the shutter trigger) stays on one thread. PTP is synchronous;
    // interleaving transactions from two threads would corrupt the pipe.
    private val captureRequested = AtomicBoolean(false)

    /** Ask the live tether loop to trigger the shutter from the host. */
    fun requestCapture() {
        captureRequested.set(true)
    }

    suspend fun run(
        onLog: (String) -> Unit,
        onCapture: suspend (filename: String, bytes: ByteArray) -> Unit,
    ) {
        val session = openAndInit(onLog) ?: run {
            onLog(
                "Couldn't start tether. Wake the camera (half-press the shutter), set the " +
                    "camera's Auto power off to Disable, confirm the USB permission was allowed, " +
                    "then tap START TETHER CAPTURE again."
            )
            return
        }
        try {
            onLog("Tether live — tap CAPTURE PHOTO to shoot.")
            val seen = HashSet<Long>()
            // Drain events already queued (props/storage) so a pre-existing object
            // isn't treated as a fresh capture.
            runCatching { EosEvents.parse(session.eosGetEvent()) }
                .getOrDefault(emptyList())
                .filter { it.isObjectArrival }
                .forEach { seen.add(it.handle) }

            var lastKeepAlive = System.currentTimeMillis()
            var lastHeartbeat = System.currentTimeMillis()
            var consecutiveErrors = 0
            // Log every distinct EOS event code we see — so a shutter press reveals
            // exactly what the R6 emits (object-arrival or something else to handle).
            val loggedCodes = HashSet<Int>()

            while (currentCoroutineContext().isActive) {
                if (captureRequested.getAndSet(false)) {
                    onLog("Triggering shutter…")
                    if (!tryShutterStrategies(session, onLog)) {
                        onLog("  every release method returned busy/error. The R6 most likely needs Live View active for a host-triggered shot (a much larger subsystem). See chat for the realistic path forward.")
                    }
                }
                val blob = try {
                    session.eosGetEvent()
                } catch (e: Exception) {
                    consecutiveErrors++
                    if (consecutiveErrors == 1 || consecutiveErrors % 25 == 0) {
                        onLog("GetEvent error (${e.message}) ×$consecutiveErrors")
                    }
                    if (consecutiveErrors >= ERRORS_BEFORE_STOP) {
                        onLog("Camera keeps rejecting GetEvent (${e.message}). Stopping — send me this log so I can adjust the EOS init.")
                        break
                    }
                    delay(POLL_MS)
                    continue
                }
                consecutiveErrors = 0

                val events = EosEvents.parse(blob)
                for (ev in events) {
                    if (loggedCodes.add(ev.code)) onLog("event 0x%04X seen".format(ev.code))
                }
                for (ev in events) {
                    if (!ev.isObjectArrival || !seen.add(ev.handle)) continue
                    onLog("Capture — event 0x%04X, handle 0x%08X".format(ev.code, ev.handle))
                    val bytes = try {
                        session.eosGetObject(ev.handle)
                    } catch (e: Exception) {
                        onLog("  download failed: ${e.message}")
                        continue
                    }
                    if (bytes.isEmpty()) {
                        onLog("  download returned 0 bytes — skipped")
                        continue
                    }
                    runCatching { session.eosTransferComplete(ev.handle) }
                    onLog("  pulled ${bytes.size / 1024} KB → queueing")
                    onCapture("R6_%08X.jpg".format(ev.handle), bytes)
                }

                val now = System.currentTimeMillis()
                if (now - lastHeartbeat > HEARTBEAT_MS) {
                    onLog("Polling… (tap CAPTURE PHOTO to shoot)")
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
            onLog("Tether session closed.")
        }
    }

    /** Open + EOS-init with retries; returns a ready session or null. */
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
                delay(SETTLE_MS)          // let the freshly claimed interface settle
                // OpenSession MUST be the first transaction. Canon EOS reuses no
                // transaction id: if GetDeviceInfo (id 0) ran first, the R6 treated
                // OpenSession (also id 0) as a retransmit — returned OK but never
                // opened a real session, so GetEvent later failed Session_Not_Open.
                val rc = s.openSession()
                onLog("OpenSession rc=0x%04X".format(rc))
                if (rc != Ptp.RC_OK && rc != Ptp.RC_SESSION_ALREADY_OPEN) {
                    throw PtpException("OpenSession 0x%04X".format(rc))
                }
                onLog("Entering EOS remote mode…")
                s.eosSetRemoteMode(1)
                s.eosSetEventMode(1)
                // NOTE: setting CaptureDestination=host pushed the R6 into a
                // permanent Device_Busy (it then expects Live View). Left unset —
                // a host-triggered shot to the default (card) is what we attempt.
                return s
            } catch (e: Exception) {
                onLog("  init attempt $attempt failed: ${e.message}")
                runCatching { s.close() }
                delay(RETRY_MS)
            }
        }
        return null
    }

    /**
     * Try the available release ops in order, stopping at the first the camera
     * ACCEPTS (no error): the simple one-shot 0x910F (no press-state, no AF
     * dependency) first, then full-press variants. "Accepted" ≠ "captured", but a
     * busy/error just moves us to the next; the poll loop catches any object event.
     */
    private suspend fun tryShutterStrategies(session: PtpSession, onLog: (String) -> Unit): Boolean {
        try {
            session.eosRemoteRelease()
            onLog("  RemoteRelease 0x910F accepted")
            return true
        } catch (e: Exception) {
            onLog("  0x910F → ${e.message}")
        }
        try {
            session.eosRemoteReleaseOn(2, 0)
            delay(300)
            session.eosRemoteReleaseOff(2)
            onLog("  RemoteReleaseOn(2) accepted")
            return true
        } catch (e: Exception) {
            onLog("  On(2) → ${e.message}")
            runCatching { session.eosRemoteReleaseOff(2) }
        }
        try {
            session.eosRemoteReleaseOn(3, 0)
            delay(300)
            session.eosRemoteReleaseOff(3)
            onLog("  RemoteReleaseOn(3) accepted")
            return true
        } catch (e: Exception) {
            onLog("  On(3) → ${e.message}")
            runCatching { session.eosRemoteReleaseOff(3) }
        }
        return false
    }

    private companion object {
        const val POLL_MS = 300L
        const val KEEPALIVE_MS = 4000L
        const val HEARTBEAT_MS = 10000L
        const val INIT_ATTEMPTS = 4
        const val RETRY_MS = 1000L
        const val SETTLE_MS = 200L
        const val ERRORS_BEFORE_STOP = 10
    }
}

/** Parser for the packed record list inside an EOS_GetEvent payload. */
object EosEvents {
    /**
     * Each record is `[UINT32 size][UINT32 eventCode][payload…]`, laid back to
     * back. For the object-arrival events the new object handle is the first
     * field of the payload (offset +8 within the record). We only need the
     * handle — GetObject's own data-phase length tells us the byte count — so we
     * don't depend on the fiddlier per-firmware size/filename offsets.
     */
    fun parse(blob: ByteArray): List<EosEvent> {
        if (blob.size < 8) return emptyList()
        val bb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
        val events = ArrayList<EosEvent>()
        var off = 0
        while (off + 8 <= blob.size) {
            val size = bb.getInt(off)
            if (size < 8) break // size 0 / terminator / malformed
            val code = bb.getInt(off + 4)
            if (code != 0 && off + 12 <= blob.size) {
                val handle = bb.getInt(off + 8).toLong() and 0xFFFFFFFFL
                events.add(EosEvent(code, handle))
            }
            if (size > blob.size - off) break
            off += size
        }
        return events
    }
}
