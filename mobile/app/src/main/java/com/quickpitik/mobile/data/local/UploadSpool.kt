package com.quickpitik.mobile.data.local

import android.content.Context
import android.os.StatFs
import android.system.ErrnoException
import android.system.OsConstants
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import java.io.File

/**
 * Where pulled camera frames wait for upload.
 *
 * Deliberately under `filesDir`, NOT `cacheDir`: Android may clear an app's
 * cache under storage pressure, and a spool file deleted before its upload
 * settles as a terminal "Local file not found" — a lost photo. Originals stay
 * here until the photographer confirms cleanup and the cloud bytes match.
 */
object UploadSpool {
    // Stop pulling frames off the camera below this much free space. Big
    // enough for the OS to keep functioning; the photographer is told to free
    // space rather than silently losing frames to a full disk.
    const val MIN_FREE_BYTES = 500L * 1024 * 1024
    const val LOW_FREE_BYTES = 1024L * 1024 * 1024
    const val STORAGE_POLL_MS = 5000L

    // Serializes camera persistence and the final cleanup check/delete, never network I/O.
    internal val filesMutex = Mutex()

    enum class Level { UNKNOWN, READY, LOW, BLOCKED }

    data class Space(val availableBytes: Long? = null) {
        val level: Level get() = when {
            availableBytes == null -> Level.UNKNOWN
            !hasRoom(CameraTransferStore.MAX_PUBLISH_BYTES) -> Level.BLOCKED
            availableBytes < LOW_FREE_BYTES -> Level.LOW
            else -> Level.READY
        }

        fun hasRoom(incomingBytes: Long): Boolean = availableBytes != null &&
            availableBytes - incomingBytes.coerceAtLeast(0) >= MIN_FREE_BYTES
    }

    fun dir(context: Context): File =
        File(context.filesDir, "upload-spool").also { it.mkdirs() }

    // filesDir and upload-spool share the volume. Use app-available bytes, excluding OS reserves.
    fun space(context: Context): Space = runCatching {
        Space(StatFs(context.filesDir.absolutePath).availableBytes)
    }.getOrDefault(Space())

    fun observeSpace(context: Context) = flow {
        while (currentCoroutineContext().isActive) {
            emit(space(context))
            delay(STORAGE_POLL_MS)
        }
    }.distinctUntilChanged().flowOn(Dispatchers.IO)

    fun hasRoom(context: Context, incomingBytes: Long = CameraTransferStore.MAX_PUBLISH_BYTES): Boolean =
        space(context).hasRoom(incomingBytes)

    fun isFull(error: Throwable): Boolean = generateSequence(error) { it.cause }.any {
        (it is ErrnoException && it.errno == OsConstants.ENOSPC) ||
            it is android.database.sqlite.SQLiteFullException ||
            it.message?.contains("ENOSPC", ignoreCase = true) == true ||
            it.message?.contains("No space left on device", ignoreCase = true) == true
    }

    fun owns(context: Context, file: File): Boolean =
        file.canonicalFile.parentFile == dir(context).canonicalFile &&
            file.name.startsWith("camera_") && file.extension == "jpg"
}
