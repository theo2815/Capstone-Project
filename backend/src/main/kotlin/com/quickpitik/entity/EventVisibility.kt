package com.quickpitik.entity

// PUBLIC events are listed on /events; UNLISTED events are reachable only by
// link (slug, share page, coverage). Only EventRepository.search filters on
// this — everything else must keep working for a link-holder.
enum class EventVisibility(val wire: String) {
    PUBLIC("public"),
    UNLISTED("unlisted");

    companion object {
        fun fromWire(raw: String?): EventVisibility? =
            entries.firstOrNull { it.wire.equals(raw?.trim(), ignoreCase = true) }
    }
}
