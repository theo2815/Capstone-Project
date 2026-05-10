package com.quickpitik.websocket

import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

@Component
class PhotoPublishedBroadcaster(
    private val sessionRegistry: EventPhotoSessionRegistry,
) {
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onPublished(event: PhotoPublishedEvent) {
        sessionRegistry.broadcast(event.eventId, event.payload)
    }
}
