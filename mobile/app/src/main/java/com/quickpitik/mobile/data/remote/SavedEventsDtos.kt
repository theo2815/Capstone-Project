package com.quickpitik.mobile.data.remote

// Mirrors the backend SavedEventSummaryDto returned by GET/POST /me/saved-events
// (backend dto/cart/SavedEventsDto.kt). The backend types `date` as LocalDate and
// `savedAt` as OffsetDateTime, but both arrive as ISO strings on the wire (same as
// EventDto.date), so they map to String here. `state` is the stored event status
// string ("ACTIVE" / "COMPLETED" / ...). The runner-facing lifecycle bucket
// (upcoming / live / open / past) is derived client-side from `date` via
// EventCatalog.deriveEventState — never read off `state`.
data class SavedEventSummaryDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val state: String,
    val bannerUrl: String?,
    val location: String,
    val savedAt: String
)

// Mirrors the backend SaveEventRequest(eventId: UUID); the UUID travels as a JSON
// string and Spring parses it back into a UUID on the server.
data class SaveEventRequest(
    val eventId: String
)
