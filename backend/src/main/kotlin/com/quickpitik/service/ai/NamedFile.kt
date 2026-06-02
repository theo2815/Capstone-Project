package com.quickpitik.service.ai

// One file part for a multi-file (mega) multipart upload to ai-api. The filename
// is sent as-is and ai-api echoes it back as the per-image `ref`, so the backend
// sets it to the photo id to map results without relying on positional order.
data class NamedFile(
    val filename: String,
    val bytes: ByteArray,
    val contentType: String,
)
