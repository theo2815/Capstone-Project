package com.quickpitik.websocket

import java.util.UUID

// Photos of one event that are now held for removal review.
data class PhotosHiddenEvent(
    val eventId: UUID,
    val photoIds: List<UUID>,
)
