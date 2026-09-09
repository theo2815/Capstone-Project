package com.quickpitik.entity

// Admin (createdBy == null) PUBLIC events are listed on /events. UNLISTED
// events — and ALL photographer-owned events, whatever their visibility — are
// reachable only by link (slug, share page, coverage) or the owner's profile.
// EventRepository.search filters on visibility AND createdBy; everything else
// must keep working for a link-holder.
enum class EventVisibility(val wire: String) {
    PUBLIC("public"),
    UNLISTED("unlisted");

    companion object {
        fun fromWire(raw: String?): EventVisibility? =
            entries.firstOrNull { it.wire.equals(raw?.trim(), ignoreCase = true) }
    }
}
