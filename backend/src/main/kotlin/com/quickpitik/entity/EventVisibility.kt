package com.quickpitik.entity

// Admin (createdBy == null) PUBLIC events are listed on /events. A photographer
// PUBLIC event is listed only when an admin turned on "Show on /events" (V89
// listedOnEvents; going UNLISTED clears it). Everything else is reachable only
// by link (slug, share page, coverage) or the owner's profile.
// EventRepository.search filters on visibility, createdBy and listedOnEvents;
// everything else must keep working for a link-holder.
enum class EventVisibility(val wire: String) {
    PUBLIC("public"),
    UNLISTED("unlisted");

    companion object {
        fun fromWire(raw: String?): EventVisibility? =
            entries.firstOrNull { it.wire.equals(raw?.trim(), ignoreCase = true) }
    }
}
