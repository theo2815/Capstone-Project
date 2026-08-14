package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import java.io.OutputStream

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
 * **A dropped link is a pause, not the end of the shoot.** A race runs hours
 * with the phone pocketed, so a knocked USB-C cable, a dozing body, or a wedged
 * PTP pipe must not silently end the session — that loses every subsequent
 * frame with no signal beyond a notification quietly disappearing. [run] holds
 * an outer loop that reopens the session and keeps going, giving up only after
 * [RECONNECT_BUDGET_MS]. Frames shot during the outage are caught up on
 * reconnect, because the dedupe set survives the session (see [run]).
 *
 * Threading contract: every transfer is a synchronous `bulkTransfer` — callers
 * MUST launch [run] on `Dispatchers.IO`. Cancellation is only observed at
 * `delay()` points; an in-flight bulkTransfer finishes or times out (≤5 s)
 * first, so stopping can lag by up to that much. The per-session `finally`
 * (event/remote mode reset + session close) is entirely non-suspending and
 * therefore safe under cancellation.
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
        onReconnecting: (Boolean) -> Unit,
        onCapture: suspend (filename: String, writeTo: (OutputStream) -> Unit) -> Boolean,
    ) {
        // These outlive any single PTP session, and that is the whole point of
        // the outer loop: on a reconnect we must NOT re-baseline. Handles shot
        // while the camera was away are absent from `seen`, so the card-diff
        // detector finds them and pulls them — the catch-up falls out for free.
        // Relaunching run() from the ViewModel instead would reset `seen`, and a
        // fresh baseline would silently swallow every frame from the outage.
        val seen = HashSet<Long>()
        val attempts = HashMap<Long, Int>()
        var baselined = false
        // When the current outage started; 0 while a session is healthy.
        var linkLostAt = 0L

        try {
            while (currentCoroutineContext().isActive) {
                val session = openAndInit(onLog)
                if (session == null) {
                    // Never got going at all: a setup problem, not a blip.
                    if (!baselined) {
                        onLog(
                            "Couldn't start auto-upload. Wake the camera (half-press the shutter), set " +
                                "Auto power off → Disable, confirm the USB permission was allowed, then " +
                                "tap Start auto-upload again."
                        )
                        return
                    }
                    if (outOfPatience(linkLostAt, onLog)) return
                    delay(RECONNECT_WAIT_MS)
                    continue
                }

                if (!baselined) {
                    if (!setBaseline(session, seen, onLog)) {
                        runCatching { session.close() }
                        return
                    }
                    baselined = true
                    onStarted()
                    onLog("Auto-upload live — press the shutter on the camera; each shot uploads.")
                } else {
                    onReconnecting(false)
                    onLog("Reconnected — catching up on anything shot while the camera was away.")
                }

                val sessionStart = System.currentTimeMillis()
                try {
                    watch(session, seen, attempts, onLog, onCapture)
                } catch (e: CancellationException) {
                    throw e // the photographer stopped — never treat this as a drop
                } catch (e: Exception) {
                    // Anything the watch loop didn't handle itself kills only
                    // this session, for the same reason the error storm does:
                    // a fresh session is the remedy, and ending the shoot over
                    // it loses every later frame.
                    onLog("Session ended on ${e.message ?: e::class.simpleName}.")
                } finally {
                    // Non-suspending, so it is safe under cancellation.
                    runCatching { session.eosSetEventMode(0) }
                    runCatching { session.eosSetRemoteMode(0) }
                    runCatching { session.close() }
                }

                if (!currentCoroutineContext().isActive) break

                // watch() returned on its own, so the link died. Only a session
                // that ran healthily for a while earns a fresh budget: a body
                // that accepts a session and immediately storms would otherwise
                // reopen forever, one short-lived session at a time.
                val wasHealthy = System.currentTimeMillis() - sessionStart >= MIN_HEALTHY_SESSION_MS
                if (wasHealthy || linkLostAt == 0L) linkLostAt = System.currentTimeMillis()
                onReconnecting(true)
                onLog("Camera link lost — retrying. Photos already pulled keep uploading.")
                if (outOfPatience(linkLostAt, onLog)) return
                delay(RECONNECT_WAIT_MS)
            }
        } finally {
            onLog("Auto-upload stopped.")
        }
    }

    /**
     * True once we have been trying to reopen for longer than
     * [RECONNECT_BUDGET_MS] — the point where "the cable got knocked" stops
     * being the likely explanation and "the photographer packed up" starts.
     */
    private fun outOfPatience(linkLostAt: Long, onLog: (String) -> Unit): Boolean {
        if (linkLostAt == 0L) return false
        if (System.currentTimeMillis() - linkLostAt < RECONNECT_BUDGET_MS) return false
        onLog(
            "Camera hasn't come back in ${RECONNECT_BUDGET_MS / 60_000} minutes — stopping " +
                "auto-upload. Photos already pulled keep uploading."
        )
        return true
    }

    /**
     * Prime BOTH detectors so frames already on the card aren't treated as new.
     * Fail-closed: without a card baseline the first scan would see every photo
     * on the card as fresh and upload all of them, so a failure here stops the
     * run rather than guessing. Runs once per [run], never on a reconnect.
     */
    private suspend fun setBaseline(
        session: PtpSession,
        seen: HashSet<Long>,
        onLog: (String) -> Unit,
    ): Boolean {
        runCatching { EosEvents.parse(session.eosGetEvent()) }.getOrDefault(emptyList())
            .filter { it.isObjectArrival }.forEach { seen.add(it.handle) }

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
            return false
        }
        seen.addAll(baseCard)
        onLog("Baseline set — ${baseCard.size} card object(s) known. Shoot now.")
        return true
    }

    /**
     * Poll one open session until it dies or the caller cancels. Returning
     * normally means "this session is finished" — the error-storm exit is a
     * reopen signal, since a fresh session is exactly the remedy for a wedged
     * PTP pipe. [seen] and [attempts] belong to [run] so they survive that.
     */
    private suspend fun watch(
        session: PtpSession,
        seen: HashSet<Long>,
        attempts: HashMap<Long, Int>,
        onLog: (String) -> Unit,
        onCapture: suspend (String, (OutputStream) -> Unit) -> Boolean,
    ) {
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
                    onLog("Camera stopped answering (${e.message}) — dropping this session.")
                    return
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
        onCapture: suspend (String, (OutputStream) -> Unit) -> Boolean,
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
        val outcome = pullInto(name, sniff = info == null, onLog, onCapture) { sink ->
            session.eosGetObjectTo(handle, sink)
        }
        // Acknowledge the object only once the bytes are durably ours. The
        // buffered version told the camera "done" before the persist; if that
        // persist then failed the frame was already released.
        if (outcome == PullOutcome.UPLOADED) runCatching { session.eosTransferComplete(handle) }
        return outcome
    }

    /** Pull a frame found by the standard card-listing diff (standard GetObject). */
    private suspend fun pullViaCard(
        session: PtpSession,
        handle: Long,
        onLog: (String) -> Unit,
        onCapture: suspend (String, (OutputStream) -> Unit) -> Boolean,
    ): PullOutcome {
        val info = runCatching { session.getObjectInfo(handle) }.getOrNull()
        if (info != null && info.isAssociation) return PullOutcome.SKIPPED
        if (info != null && !info.isJpeg) return PullOutcome.SKIPPED // RAW stays on the card
        val name = info?.filename?.takeIf { it.isNotBlank() } ?: "IMG_%08X.jpg".format(handle)
        onLog("Capture via CARD — $name (0x%08X)".format(handle))
        return pullInto(name, sniff = info == null, onLog, onCapture) { sink ->
            session.getObjectTo(handle, sink)
        }
    }

    /**
     * Shared tail of both detectors: stream the object into the caller's sink
     * and fold the result into a [PullOutcome].
     *
     * [sniff] is set only when GetObjectInfo failed and there is no format to
     * filter on, so the JPEG SOI marker has to be checked on the wire. The sink
     * aborts on the first two bytes, which means a stray RAW now costs two
     * bytes instead of a full-size download.
     *
     * That abort surfaces as `persisted == false` rather than as a throw here:
     * the exception is raised inside the caller's sink handling (which owns
     * deleting the partial file), so the outcome is read back off the sniffer.
     */
    private suspend fun pullInto(
        name: String,
        sniff: Boolean,
        onLog: (String) -> Unit,
        onCapture: suspend (String, (OutputStream) -> Unit) -> Boolean,
        read: (OutputStream) -> Long,
    ): PullOutcome {
        var pulled = 0L
        var sniffer: JpegSniffSink? = null
        val persisted = try {
            onCapture(name) { sink ->
                val target = if (sniff) JpegSniffSink(sink).also { sniffer = it } else sink
                pulled = read(target)
            }
        } catch (e: Exception) {
            onLog("  download failed: ${e.message}")
            return PullOutcome.FAILED
        }
        if (sniffer?.sawNonJpegHead == true) {
            // A CR3 never becomes a JPEG, so this is a skip, not a retry.
            onLog("  not a JPEG (no SOI) — left on card")
            return PullOutcome.SKIPPED
        }
        if (!persisted) {
            // 0 bytes usually means the object is still being written to the
            // card; either way it is worth another try when a detector
            // surfaces the handle again.
            onLog("  ${if (pulled == 0L) "0 bytes" else "persist failed"} — will retry")
            return PullOutcome.FAILED
        }
        onLog("  $name ${pulled / 1024} KB → queued")
        return PullOutcome.UPLOADED
    }

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

        // Reconnect budget. A race lasts hours and a knocked cable re-enumerates
        // in seconds, so patience is cheap; five minutes covers a lens change or
        // a body that dozed off, and still ends the shoot in a bounded time once
        // the photographer has actually packed up.
        const val RECONNECT_BUDGET_MS = 5L * 60L * 1000L
        const val RECONNECT_WAIT_MS = 2000L

        // A session that dies faster than this never really worked, so it does
        // NOT refresh the budget above — otherwise a body that accepts a session
        // and immediately storms would reopen forever, one short session at a time.
        const val MIN_HEALTHY_SESSION_MS = 10_000L
    }
}

/**
 * Passes bytes straight through to [delegate] while checking that the object
 * begins with a JPEG SOI marker (FF D8).
 *
 * Only used when GetObjectInfo failed and there is no declared format to filter
 * on. Aborting on the first two bytes is the point: with the buffered reads this
 * check happened after a whole RAW had already crossed the wire.
 */
private class JpegSniffSink(private val delegate: OutputStream) : OutputStream() {

    /** True once the head was inspected and did not look like a JPEG. */
    var sawNonJpegHead = false
        private set

    private val head = ByteArray(2)
    private var headLen = 0
    private var verified = false

    override fun write(b: Int) = write(byteArrayOf(b.toByte()), 0, 1)

    override fun write(b: ByteArray, off: Int, len: Int) {
        var i = 0
        while (headLen < head.size && i < len) {
            head[headLen] = b[off + i]
            headLen++
            i++
        }
        if (!verified && headLen == head.size) {
            verified = true
            if (head[0] != SOI_0 || head[1] != SOI_1) {
                sawNonJpegHead = true
                throw PtpException("object does not start with a JPEG SOI marker")
            }
        }
        delegate.write(b, off, len)
    }

    private companion object {
        const val SOI_0 = 0xFF.toByte()
        const val SOI_1 = 0xD8.toByte()
    }
}
