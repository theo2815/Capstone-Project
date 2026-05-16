package com.quickpitik.service.admin

import java.time.ZoneId

internal val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")

// Event → AdminListEventDto lives in EventDtoMapper (events package) — it
// needs StorageService to presign cover_s3_key, which AdminMappers does not
// have. The admin-state derive + city-from-location helpers also live in
// EventDtoMapper.companion so both event and admin surfaces share one
// implementation.

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
