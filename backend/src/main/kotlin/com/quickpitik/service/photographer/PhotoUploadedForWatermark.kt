package com.quickpitik.service.photographer

import java.util.UUID

// Published AFTER_COMMIT of a photo upload so PhotoWatermarkTrigger can
// generate the watermark derivative asynchronously, off the request thread,
// and flip the photo PROCESSING → LIVE. Unlike PhotoUploadedForIndexing this
// is NOT gated on ai-api — watermarking always runs.
data class PhotoUploadedForWatermark(
    val photoId: UUID,
    val eventId: UUID,
)
