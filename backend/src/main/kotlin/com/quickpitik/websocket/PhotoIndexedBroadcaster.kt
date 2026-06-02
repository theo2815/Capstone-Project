package com.quickpitik.websocket

import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

@Component
class PhotoIndexedBroadcaster(
    private val sessionRegistry: EventPhotoSessionRegistry,
) {
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onIndexed(event: PhotoIndexedEvent) {
        sessionRegistry.broadcast(event.eventId, event.payload)
    }
}
