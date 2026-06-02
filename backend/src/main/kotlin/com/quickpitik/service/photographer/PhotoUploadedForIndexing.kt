package com.quickpitik.service.photographer

import java.util.UUID

// Published AFTER_COMMIT of a photo upload so PhotoIndexingTrigger can run
// face enroll + bib OCR asynchronously, off the request thread.
data class PhotoUploadedForIndexing(
    val photoId: UUID,
    val eventId: UUID,
)
