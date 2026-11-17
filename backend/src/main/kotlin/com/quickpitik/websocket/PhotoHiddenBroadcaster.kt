package com.quickpitik.websocket

import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

@Component
class PhotoHiddenBroadcaster(
    private val sessionRegistry: EventPhotoSessionRegistry,
) {
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onHidden(event: PhotosHiddenEvent) {
        event.photoIds.forEach { sessionRegistry.broadcast(event.eventId, mapOf("type" to "photo.hidden", "photoId" to it)) }
    }
}
