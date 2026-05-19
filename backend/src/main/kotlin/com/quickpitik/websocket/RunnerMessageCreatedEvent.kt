package com.quickpitik.websocket

import java.util.UUID

// Published by RunnerMessagesService.pushMessage after the inbox row is
// saved. Broadcast by NotificationBroadcaster on AFTER_COMMIT — rollback
// of the surrounding admin TX must NOT push a phantom message to the
// runner's bell.
//
// `payload` mirrors the FE RunnerMessageDto wire shape so the bell can
// merge it directly into the messages query cache without a refetch.
data class RunnerMessageCreatedEvent(
    val runnerId: UUID,
    val payload: Map<String, Any?>,
)
