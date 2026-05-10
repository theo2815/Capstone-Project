package com.quickpitik.websocket

import java.util.UUID

data class PhotoPublishedEvent(
    val eventId: UUID,
    val payload: Map<String, Any?>,
)
