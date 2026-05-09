package com.quickpitik.service.storage

import java.io.InputStream
import java.time.Duration

interface StorageService {
    fun put(key: String, bytes: ByteArray, contentType: String): StoredObject

    fun put(key: String, stream: InputStream, contentLength: Long, contentType: String): StoredObject

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
