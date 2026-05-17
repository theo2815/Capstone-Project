package com.quickpitik.websocket

import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

// AFTER_COMMIT broadcaster for the notification WebSocket channels. Both
// listeners run after the surrounding admin/photographer transaction
// commits — a rollback discards the inbox row AND silently drops the
// would-be push so the FE never sees a phantom badge.
//
// Same pattern as PhotoPublishedBroadcaster (Q-002) — the existing
// rationale applies verbatim: inline broadcast risks ghost data the FE
// then 404s on.
@Component
class NotificationBroadcaster(
    private val meRegistry: MeNotificationSessionRegistry,
    private val adminRegistry: AdminNotificationSessionRegistry,
) {
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onPhotographerMessage(event: PhotographerMessageCreatedEvent) {
        meRegistry.broadcast(event.photographerId, event.payload)
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onAdminInbox(event: AdminInboxEvent) {
        adminRegistry.broadcast(event.payload)
    }
}
