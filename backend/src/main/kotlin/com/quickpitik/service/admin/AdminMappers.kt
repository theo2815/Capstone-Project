package com.quickpitik.service.admin

import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import java.time.LocalDate
import java.time.ZoneId

internal val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")

// Mirrors website/src/app/events/events-browser.tsx EventState mapping plus
// the photographer-side derive in PhotographerEventDtos. Defined here too
// so admin can use it without a backwards dependency on the photographer
// package (clean separation by domain).
internal fun deriveAdminEventState(event: Event, today: LocalDate = LocalDate.now(PH_ZONE)): String =
    when (event.status) {
        EventStatus.ACTIVE -> when {
            event.date.isEqual(today) -> "live"
            event.date.isAfter(today) -> "upcoming"
            else -> "open"
        }
        EventStatus.COMPLETED -> "open"
        EventStatus.ARCHIVED -> "past"
        // Drafts aren't part of the public state machine — admin sees them
        // as "upcoming" so the row keeps a sensible visual shape until the
        // admin promotes the row to ACTIVE.
        EventStatus.DRAFT -> "upcoming"
    }

// FE expects the city as the part after the comma in `location`. Mirrors the
// public `cityFromLocation` helper used elsewhere on the runner-facing
// surfaces; admin parses the same shape so /admin/events tiles render.
internal fun cityFromLocation(location: String): String {
    val idx = location.lastIndexOf(", ")
    return if (idx >= 0) location.substring(idx + 2).trim() else location.trim()
}

internal fun Event.toAdminListDto(): AdminListEventDto =
    AdminListEventDto(
        id = id,
        slug = slug,
        name = name,
        date = date,
        location = location,
        bannerUrl = bannerUrl,
        photoCount = photoCount,
        participantCount = participantCount,
        status = status.name,
        state = deriveAdminEventState(this),
        city = cityFromLocation(location),
        pricePerPhoto = pricePerPhoto,
        description = description,
        organizerName = organizerName,
        categories = categories.sorted(),
    )

// Cebu · Central Visayas style label. Codes look like "region-7" / "cebu";
// the json source has names like "Region VII (Central Visayas)" + province
// "Cebu". The FE expects just "<Province> · <RegionShort>". This helper
// builds it from the resolved names.
internal fun formatRegionLabel(provinceName: String?, regionName: String?): String? {
    if (provinceName.isNullOrBlank()) return null
    val regionShort = shortenRegionName(regionName.orEmpty())
    return if (regionShort.isBlank()) provinceName else "$provinceName · $regionShort"
}

// "Region VII (Central Visayas)" → "Central Visayas". The parenthetical is
// the friendly name the FE renders; falling back to the raw name covers
// regions like NCR / BARMM that don't follow the same pattern.
private fun shortenRegionName(name: String): String {
    val open = name.indexOf('(')
    val close = name.indexOf(')')
    if (open >= 0 && close > open) return name.substring(open + 1, close).trim()
    return name.trim()
}

// Privacy-safe handle for runner display in disputes. The FE seed uses the
// runner's first-name ("juan", "thea") so we mirror that — drop spaces +
// lowercase the first word of `User.name`. Falls back to email local-part
// when name is blank.
internal fun runnerDisplayHandle(name: String, email: String?): String {
    val firstWord = name.trim().split(Regex("\\s+")).firstOrNull().orEmpty()
    if (firstWord.isNotEmpty()) return firstWord.lowercase()
    val local = email?.substringBefore('@').orEmpty().trim()
    if (local.isNotEmpty()) return local.lowercase()
    return ""
}
