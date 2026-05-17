package com.quickpitik.websocket

import java.util.UUID

// Published by AdminDecisionLogService.pushMessage after the inbox row is
// saved. Broadcast by NotificationBroadcaster on AFTER_COMMIT — rollback
// of the surrounding admin TX must NOT push a phantom message to the
// photographer's bell.
//
// `payload` mirrors the PhotographerMessageDto wire shape so the FE can
// merge it directly into useMyPhotographerMessagesStore without a refetch.
data class PhotographerMessageCreatedEvent(
    val photographerId: UUID,
    val payload: Map<String, Any?>,
)
