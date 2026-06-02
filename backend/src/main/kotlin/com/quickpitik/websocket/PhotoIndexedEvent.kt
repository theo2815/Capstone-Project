package com.quickpitik.websocket

import java.util.UUID

// Fired AFTER_COMMIT when async indexing finishes a photo (face + bib tags
// written). Rides the same event-photo WebSocket channel as photo.published.
data class PhotoIndexedEvent(
    val eventId: UUID,
    val payload: Map<String, Any?>,
)
