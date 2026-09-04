package com.quickpitik.mobile.data.local

import android.content.Context
import java.io.File

/**
 * Where pulled camera frames wait for upload.
 *
 * Deliberately under `filesDir`, NOT `cacheDir`: Android may clear an app's
 * cache under storage pressure, and a spool file deleted before its upload
 * settles as a terminal "Local file not found" — a lost photo. Nothing here
 * outlives its usefulness anyway: PhotoUploadWorker shrinks each file to a
 * ~20 KB thumbnail the moment the upload completes.
 */
object UploadSpool {
    // Stop pulling frames off the camera below this much free space. Big
    // enough for the OS to keep functioning; the photographer is told to free
    // space rather than silently losing frames to a full disk.
    const val MIN_FREE_BYTES = 500L * 1024 * 1024

    fun dir(context: Context): File =
        File(context.filesDir, "upload-spool").also { it.mkdirs() }

    fun hasRoom(context: Context): Boolean = dir(context).usableSpace >= MIN_FREE_BYTES
}
