package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive

/**
 * One-shot enumeration of the JPEGs already on the camera's memory card — the
 * read half of the manual "Import from camera" flow.
 *
 * Unlike [UsbCardPollController] (which holds a session open and watches for
 * NEW shots), this opens a PTP session, asks the camera for everything on the
 * card right now, closes the session, and returns. The Capture tab calls this
 * after the photographer plugs the body in post-event — it gives them a list
 * they can select from and import in bulk. The poll controller stays in the
 * codebase for the future live-shoot path; the two are intentionally distinct.
 *
 * The session is always closed in finally — leaving it open would freeze the
 * R6's physical shutter (documented Canon "PC connection" behaviour). The
 * single round-trip-per-handle for [PtpSession.getObjectInfo] is what makes
 * the scan visible enough to report progress.
 */
class UsbCardBrowseController(
    private val manager: UsbManager,
    private val deviceProvider: () -> UsbDevice?,
) {
    /** Streamed lifecycle of a single browse run — the VM maps these to UI state. */
    sealed class Progress {
        /** Opening the session — first phase, usually finishes in <1s. */
        data object Opening : Progress()

        /** Walking the card; [seen] of [total] handles inspected so far. */
        data class Scanning(val seen: Int, val total: Int) : Progress()

        /**
         * All handles inspected — [photos] are the JPEGs we'd offer for import,
         * WITHOUT thumbnails yet. The list is usable (select / import) at once.
         */
        data class Done(val photos: List<CardPhoto>) : Progress()

        /**
         * Same list with thumbnails filled in so far. Emitted in batches after
         * [Done] while the session stays open, then once more at the end. A
         * 200-frame card used to sit behind a ~30 s blank scan because thumbs
         * were fetched before anything was shown.
         */
        data class Thumbnails(val photos: List<CardPhoto>) : Progress()

        /** A user-presentable failure message — no raw exception strings. */
        data class Failed(val message: String) : Progress()
    }

    suspend fun browse(emit: suspend (Progress) -> Unit) {
        emit(Progress.Opening)

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
        } catch (e: PtpException) {
            emit(Progress.Failed(MSG_OPEN_FAILED))
            return
        } catch (e: Exception) {
            emit(Progress.Failed(MSG_OPEN_FAILED))
            return
        }

        try {
            val handles = collectHandles(session)
            if (handles.isEmpty()) {
                emit(Progress.Done(emptyList()))
                return
            }

            val photos = ArrayList<CardPhoto>()
            for ((idx, handle) in handles.withIndex()) {
                if (!currentCoroutineContext().isActive) return
                val info = runCatching { session.getObjectInfo(handle) }.getOrNull()
                // Folders and RAW stay off the list — only JPEGs are uploadable.
                if (info != null && !info.isAssociation && info.isJpeg) {
                    photos.add(
                        CardPhoto(
                            handle = handle,
                            filename = info.filename.ifBlank { defaultName(handle) },
                            sizeBytes = info.compressedSize,
                        )
                    )
                }
                emit(Progress.Scanning(seen = idx + 1, total = handles.size))
            }
            emit(Progress.Done(photos))

            // Progressive thumbnails (Increment 5, now after the list is shown).
            // Best-effort per handle — some bodies don't expose GetThumb for
            // every file; a miss keeps the placeholder tile. Newest-first order
            // means the frames the photographer actually wants fill in first.
            val withThumbs = photos.toMutableList()
            for ((i, photo) in photos.withIndex()) {
                if (!currentCoroutineContext().isActive) return
                val thumb = runCatching { session.getThumb(photo.handle) }
                    .getOrNull()
                    ?.takeIf { it.isNotEmpty() }
                    ?: continue
                withThumbs[i] = photo.copy(thumbnailBytes = thumb)
                if ((i + 1) % THUMB_BATCH == 0) emit(Progress.Thumbnails(withThumbs.toList()))
            }
            emit(Progress.Thumbnails(withThumbs.toList()))
        } catch (e: Exception) {
            emit(Progress.Failed(MSG_READ_FAILED))
        } finally {
            runCatching { session.close() }
        }
    }

    /** Union of object handles across every storage, with the wildcard fallback. */
    private fun collectHandles(session: PtpSession): List<Long> {
        val all = LinkedHashSet<Long>()
        for (sid in runCatching { session.getStorageIds() }.getOrDefault(emptyList())) {
            runCatching { session.getObjectHandles(sid) }.getOrNull()?.let(all::addAll)
        }
        if (all.isEmpty()) {
            // Some bodies prefer the "all stores" wildcard (0xFFFFFFFF).
            runCatching { session.getObjectHandles(0xFFFFFFFFL) }.getOrNull()?.let(all::addAll)
        }
        // Canon increments object handles — ascending = capture order, so
        // descending puts the freshly-shot photos at the TOP of the import
        // list. That's what the photographer expects after a session: the
        // last 5 frames they remember taking are visible without scrolling
        // through 200 earlier shots. Selection / dedupe / import are all
        // keyed by handle, so the reorder is purely cosmetic.
        return all.sortedDescending()
    }

    private fun defaultName(handle: Long): String = "IMG_%08X.jpg".format(handle)

    private companion object {
        const val THUMB_BATCH = 8
        const val MSG_NO_DEVICE =
            "No camera detected. Re-plug the USB cable and allow USB permission, then try again."
        const val MSG_OPEN_FAILED =
            "Couldn't open the camera. Set its USB mode to PC connection / PTP " +
                "(\"Photo Import\") and re-plug the cable."
        const val MSG_SESSION_REJECTED =
            "Camera refused to open a session. Detach any other PC connection on the body, then try again."
        const val MSG_READ_FAILED =
            "Card read interrupted. Check the cable and that the camera stayed awake, then try again."
    }
}

/**
 * One JPEG on the card, as far as the browser cares: a PTP object handle,
 * filename, compressed size, and an optional camera-rendered thumbnail JPEG
 * (Increment 5 — null when the camera refused GetThumb for that handle).
 * Full bytes are pulled separately during import (Increment 3).
 */
data class CardPhoto(
    val handle: Long,
    val filename: String,
    val sizeBytes: Long,
    val thumbnailBytes: ByteArray? = null,
) {
    // ByteArray's default equals/hashCode are identity-based, which breaks
    // LazyColumn diffing if a CardPhoto ever gets reconstructed. Override to
    // make `data class` semantics stable — we only ever compare by handle in
    // practice anyway (it's the LazyColumn key).
    override fun equals(other: Any?): Boolean =
        other is CardPhoto && handle == other.handle
    override fun hashCode(): Int = handle.hashCode()
}
