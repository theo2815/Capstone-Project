package com.quickpitik.service.storage

import org.springframework.http.MediaType

/**
 * Content type for a stored object, derived from its key. Storage keeps no
 * metadata we trust for this, and both face-search entry points plus the
 * photos-ready notifier need the same answer for the same selfie key.
 */
fun contentTypeForKey(key: String): String = when (key.substringAfterLast('.').lowercase()) {
    "jpg", "jpeg" -> MediaType.IMAGE_JPEG_VALUE
    "png" -> MediaType.IMAGE_PNG_VALUE
    "webp" -> "image/webp"
    else -> MediaType.APPLICATION_OCTET_STREAM_VALUE
}
