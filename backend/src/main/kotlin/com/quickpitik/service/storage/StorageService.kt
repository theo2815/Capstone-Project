package com.quickpitik.service.storage

import java.io.InputStream
import java.time.Duration

interface StorageService {
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

    fun presignedPutUrl(key: String, ttl: Duration, contentType: String): String
}

data class StoredObject(
    val key: String,
    val sizeBytes: Long,
    val contentType: String,
)
