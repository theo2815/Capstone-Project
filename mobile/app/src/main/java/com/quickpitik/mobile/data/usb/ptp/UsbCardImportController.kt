package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import java.io.OutputStream

/**
 * Pulls the JPEG bytes for a chosen subset of card handles — the transfer half
 * of the manual "Import from camera" flow (Increment 3).
 *
 * Same lifecycle as [UsbCardBrowseController]: open a PTP session, do the work,
 * close it in finally. Holding the session open between selection and transfer
 * isn't worth it — the R6 locks its physical shutter while a session is open,
 * so two short sessions (browse, then import) beats one long one. The decision
 * mirrors the 2026-05-26 Zno-pivot ADR for the same reason.
 *
 * The bytes don't land in this class — the caller's [onPulled] callback writes
 * them to whatever store the VM owns (today: app cacheDir + Room upload queue).
 * That keeps the controller free of Android persistence APIs and trivially
 * testable.
 */
class UsbCardImportController(
    private val manager: UsbManager,
    private val deviceProvider: () -> UsbDevice?,
) {
    /**
     * Streamed lifecycle of an import run. Per-handle Set-of-Long tallies (not
     * just counts) so the VM can power both retry-failed (Increment 4 B) and
     * cross-run dedupe (Increment 4 D).
     */
    sealed class Progress {
        /** Open succeeded; about to start pulling [total] photos. */
        data class Started(val total: Int) : Progress()

        /** One photo finished (success or fail). Running sets for the UI. */
        data class Each(
            val seen: Int,
            val total: Int,
            val succeededHandles: Set<Long>,
            val failedHandles: Set<Long>,
        ) : Progress()

        /** All photos attempted — final tallies; session is already closed. */
        data class Done(
            val succeededHandles: Set<Long>,
            val failedHandles: Set<Long>,
        ) : Progress()

        /** Session-level failure (open / fatal read). Per-photo failures don't get here. */
        data class Failed(val message: String) : Progress()
    }

    /**
     * Pull each photo in [photos] in order. For each one, [onPulled] is invoked
     * with a writer that streams the JPEG into whatever sink the caller opens,
     * and should return true if the bytes were durably persisted (file written
     * + queued for upload). A return of false counts the photo as failed so the
     * UI summary stays honest.
     *
     * The bytes are handed over as a writer rather than a `ByteArray` so a large
     * frame never exists in memory: the PTP data phase lands directly in the
     * caller's file. Exceptions thrown by the writer surface inside the caller's
     * sink handling, which is where the partial file has to be cleaned up.
     *
     * Per-photo failures (PTP read errors, persist errors) are non-fatal — the
     * loop carries on and reports the tally. Session-level failures (no device,
     * open failed, the read loop throws) end the run with [Progress.Failed].
     */
    suspend fun import(
        photos: List<CardPhoto>,
        emit: suspend (Progress) -> Unit,
        onPulled: suspend (photo: CardPhoto, writeTo: (OutputStream) -> Unit) -> Boolean,
    ) {
        if (photos.isEmpty()) {
            emit(Progress.Done(succeededHandles = emptySet(), failedHandles = emptySet()))
            return
        }

        val device = deviceProvider() ?: run {
            emit(Progress.Failed(MSG_NO_DEVICE))
            return
        }

        val session = try {
            PtpSession(manager, device).also { s ->
                val rc = s.openSession()
                if (rc != Ptp.RC_OK && rc != Ptp.RC_SESSION_ALREADY_OPEN) {
                    runCatching { s.close() }
                    emit(Progress.Failed(MSG_SESSION_REJECTED))
                    return
                }
            }
        } catch (e: Exception) {
            emit(Progress.Failed(MSG_OPEN_FAILED))
            return
        }

        emit(Progress.Started(total = photos.size))

        try {
            val succeeded = LinkedHashSet<Long>()
            val failed = LinkedHashSet<Long>()

            for ((idx, photo) in photos.withIndex()) {
                if (!currentCoroutineContext().isActive) return

                // Timed inside the writer so the log keeps reporting the PTP
                // read alone, not the caller's disk write on top of it.
                var pulledBytes = 0L
                var ptpMs = 0L
                val didPersist = runCatching {
                    onPulled(photo) { sink ->
                        val ptpStart = System.currentTimeMillis()
                        pulledBytes = session.getObjectTo(photo.handle, sink)
                        ptpMs = System.currentTimeMillis() - ptpStart
                    }
                }.getOrDefault(false)

                // QP/UPLOAD-PERF — `adb logcat -s QP/UPLOAD-PERF` shows
                // per-phase ms for each photo (ptp / persist / upload).
                android.util.Log.i(
                    "QP/UPLOAD-PERF",
                    "ptp-read handle=${photo.handle} file=${photo.filename} bytes=$pulledBytes ms=$ptpMs",
                )

                if (didPersist) succeeded.add(photo.handle) else failed.add(photo.handle)

                emit(
                    Progress.Each(
                        seen = idx + 1,
                        total = photos.size,
                        succeededHandles = succeeded.toSet(),
                        failedHandles = failed.toSet(),
                    )
                )
            }
            emit(
                Progress.Done(
                    succeededHandles = succeeded.toSet(),
                    failedHandles = failed.toSet(),
                )
            )
        } catch (e: Exception) {
            // Fatal — surface as session-level failure (the UI shows Error,
            // not a partial Done) so the user knows to retry, not just close.
            emit(Progress.Failed(MSG_READ_FAILED))
        } finally {
            runCatching { session.close() }
        }
    }

    private companion object {
        const val MSG_NO_DEVICE =
            "Camera no longer detected. Re-plug the USB cable and try again."
        const val MSG_OPEN_FAILED =
            "Couldn't open the camera. Set its USB mode to PC connection / PTP " +
                "(\"Photo Import\") and re-plug the cable."
        const val MSG_SESSION_REJECTED =
            "Camera refused to open a session. Detach any other PC connection on the body, then try again."
        const val MSG_READ_FAILED =
            "Card transfer interrupted. Check the cable and that the camera stayed awake."
    }
}
