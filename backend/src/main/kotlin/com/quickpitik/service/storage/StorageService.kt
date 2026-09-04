package com.quickpitik.service.storage

import java.io.InputStream
import java.time.Duration

interface StorageService {
    // True when clients can PUT straight to storage with presignedPutUrl —
    // S3/R2. The local-disk dev backend serves GET only, so photo uploads
    // fall back to the multipart endpoint there.
    val supportsDirectUpload: Boolean

    fun put(key: String, bytes: ByteArray, contentType: String): StoredObject

    fun put(key: String, stream: InputStream, contentLength: Long, contentType: String): StoredObject

    // Server-side read of stored object bytes. Used when a server-internal
    // operation (e.g. compositing the photographer's watermark image onto a
    // newly-uploaded photo) needs the bytes, not a presigned URL the client
    // would fetch. Throws if the key is missing.
    fun getBytes(key: String): ByteArray

    fun delete(key: String)

    fun exists(key: String): Boolean

    fun presignedGetUrl(key: String, ttl: Duration): String

    // Download variant: the response carries
    //   Content-Disposition: attachment; filename="…"
    // so cross-origin clicks save instead of displaying inline (<a download>
    // is ignored cross-origin). On S3/R2 the disposition must be part of the
    // SIGNED query — params appended after presigning 403 under SigV4 —
    // which is why clients can't do this themselves. `filename` must already
    // be header-safe ASCII (callers sanitize; see OrderService).
    fun presignedDownloadUrl(key: String, ttl: Duration, filename: String): String

    fun presignedPutUrl(key: String, ttl: Duration, contentType: String): String
}

data class StoredObject(
    val key: String,
    val sizeBytes: Long,
    val contentType: String,
)
