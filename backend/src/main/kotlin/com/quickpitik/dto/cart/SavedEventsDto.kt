package com.quickpitik.dto.cart

import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

data class SaveEventRequest(val eventId: UUID)

data class RemovedResponse(val removed: Boolean)

data class ClearedResponse(val cleared: Int)

data class MergeSavedEventsRequest(val eventIds: List<UUID> = emptyList())

// 2026-05-19 PM — GET /me/saved-events + POST /me/saved-events + merge now
// return full SavedEventSummaryDto rows instead of bare UUIDs. The /profile
// race log renders directly off these (name + date + state + cover) without
// needing a second round-trip per saved event. SaveEventResponse removed —
// POST returns the same SavedEventSummaryDto so the FE can optimistic-insert
// without waiting for a list refetch.
data class SavedEventSummaryDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val state: String,
    val bannerUrl: String?,
    val location: String,
    val savedAt: OffsetDateTime,
)
