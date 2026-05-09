package com.quickpitik.dto.cart

import java.time.OffsetDateTime
import java.util.UUID

data class SaveEventRequest(val eventId: UUID)

data class SaveEventResponse(val savedAt: OffsetDateTime)

data class RemovedResponse(val removed: Boolean)

data class ClearedResponse(val cleared: Int)

data class MergeSavedEventsRequest(val eventIds: List<UUID> = emptyList())
