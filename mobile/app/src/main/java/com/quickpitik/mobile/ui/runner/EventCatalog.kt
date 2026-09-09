package com.quickpitik.mobile.ui.runner

import com.quickpitik.mobile.data.remote.EventDto
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit

// Mirrors the website's event-catalog.ts deriveEventState(): the lifecycle stage a
// runner sees is computed CLIENT-SIDE from the event date, not the stored status.
// Keeping the same buckets here means mobile segments events identically to /events
// on the web (upcoming / live / open / past), so both surfaces agree day to day.
//
// Day math (anchored to local midnight):
//   days = today - eventDate  →  event today = 0, future < 0, past > 0
//   upcoming  days < 0
//   live      days == 0L  (race day strictly)
//   open      1..89       (recent, photos all in, still searchable)
//   past      >= 90       (archive)
enum class EventState { UPCOMING, LIVE, OPEN, PAST }

fun deriveEventState(date: String, today: LocalDate = LocalDate.now()): EventState {
    val eventDate = runCatching { LocalDate.parse(date) }.getOrNull() ?: return EventState.OPEN
    val days = ChronoUnit.DAYS.between(eventDate, today)
    return when {
        days < 0L -> EventState.UPCOMING
        days == 0L -> EventState.LIVE
        days <= 89L -> EventState.OPEN
        else -> EventState.PAST
    }
}

// The web extracts the city from the tail of "Venue, City" via lastIndexOf(",").
fun extractCity(location: String): String {
    val idx = location.lastIndexOf(',')
    return if (idx == -1) location.trim() else location.substring(idx + 1).trim()
}

// ─── Philippine Regions Reference Data ───────────────────────────────────────
data class PhRegion(
    val code: String,
    val name: String,
    val shortName: String,
    val keywords: List<String>
)

val PH_REGIONS: List<PhRegion> = listOf(
    PhRegion("ncr", "NCR (Metro Manila)", "NCR", listOf("manila", "quezon city", "taguig", "bgc", "makati", "pasig", "pasay", "mandaluyong", "paranaque", "las pinas", "muntinlupa", "marikina", "valenzuela", "malabon", "navotas", "san juan", "pateros", "caloocan", "ncr", "metro manila")),
    PhRegion("car", "Cordillera (CAR)", "CAR", listOf("benguet", "baguio", "abra", "apayao", "ifugao", "kalinga", "mountain province", "car", "cordillera")),
    PhRegion("region-1", "Region I (Ilocos)", "Region I", listOf("ilocos", "la union", "pangasinan", "dagupan", "san fernando", "vigan", "laoag")),
    PhRegion("region-2", "Region II (Cagayan Valley)", "Region II", listOf("batanes", "cagayan", "tuguegarao", "isabela", "nueva vizcaya", "quirino")),
    PhRegion("region-3", "Region III (Central Luzon)", "Region III", listOf("pampanga", "angeles", "clark", "bulacan", "bataan", "subic", "zambales", "tarlac", "nueva ecija", "aurora", "central luzon")),
    PhRegion("region-4a", "Region IV-A (CALABARZON)", "Region IV-A", listOf("cavite", "tagaytay", "laguna", "santa rosa", "calamba", "batangas", "lipa", "quezon", "lucena", "rizal", "antipolo", "calabarzon")),
    PhRegion("region-4b", "Region IV-B (MIMAROPA)", "Region IV-B", listOf("palawan", "puerto princesa", "coron", "el nido", "oriental mindoro", "occidental mindoro", "marinduque", "romblon", "mimaropa")),
    PhRegion("region-5", "Region V (Bicol)", "Region V", listOf("albay", "legazpi", "camarines sur", "naga", "camarines norte", "catanduanes", "masbate", "sorsogon", "bicol")),
    PhRegion("region-6", "Region VI (Western Visayas)", "Region VI", listOf("iloilo", "negros occidental", "bacolod", "aklan", "boracay", "antique", "capiz", "roxas", "guimaras", "western visayas")),
    PhRegion("region-7", "Region VII (Central Visayas)", "Region VII", listOf("cebu", "mandaue", "lapu-lapu", "bohol", "tagbilaran", "negros oriental", "dumaguete", "siquijor", "central visayas")),
    PhRegion("region-8", "Region VIII (Eastern Visayas)", "Region VIII", listOf("leyte", "tacloban", "ormoc", "southern leyte", "samar", "eastern samar", "northern samar", "biliran", "eastern visayas")),
    PhRegion("region-9", "Region IX (Zamboanga)", "Region IX", listOf("zamboanga", "pagadian", "dipolog", "sibugay")),
    PhRegion("region-10", "Region X (Northern Mindanao)", "Region X", listOf("cagayan de oro", "cdo", "misamis oriental", "misamis occidental", "bukidnon", "malaybalay", "camiguin", "iligan", "lanao del norte", "northern mindanao")),
    PhRegion("region-11", "Region XI (Davao Region)", "Region XI", listOf("davao", "tagum", "panabo", "mati", "davao de oro", "davao del norte", "davao del sur", "davao oriental", "davao occidental")),
    PhRegion("region-12", "Region XII (SOCCSKSARGEN)", "Region XII", listOf("general santos", "gensan", "south cotabato", "koronadal", "cotabato", "sarangani", "sultan kudarat", "soccsksargen")),
    PhRegion("region-13", "Region XIII (Caraga)", "Region XIII", listOf("butuan", "surigao", "siargao", "agusan del norte", "agusan del sur", "dinagat", "caraga")),
    PhRegion("barmm", "BARMM (Bangsamoro)", "BARMM", listOf("barmm", "bangsamoro", "cotabato city", "basilan", "sulu", "tawi-tawi", "maguindanao", "lanao del sur", "marawi"))
)

/**
 * Resolves an event location string (e.g. "Cebu Business Park, Cebu City", "BGC, Taguig")
 * to its standard Philippine administrative region display name.
 */
fun resolveRegion(location: String): String {
    val lower = location.lowercase()
    for (region in PH_REGIONS) {
        if (region.keywords.any { lower.contains(it) }) {
            return region.name
        }
    }
    val city = extractCity(location)
    return if (city.isNotBlank()) city else "Other Regions"
}

private val DATE_LABEL_FORMATTER = DateTimeFormatter.ofPattern("MMM d, yyyy")

fun eventDateLabel(date: String): String =
    runCatching { LocalDate.parse(date).format(DATE_LABEL_FORMATTER).uppercase() }.getOrDefault(date)

// Lifecycle stage paired with the event, computed once so filtering/segmenting reuses it.
fun EventDto.lifecycleState(today: LocalDate = LocalDate.now()): EventState = deriveEventState(date, today)

// Photographer upload window: race-day + 3-day grace window (days 0..3).
// Backend enforces the same window server-side as EVENT_NOT_UPLOADABLE.
fun canUploadToEvent(date: String, today: LocalDate = LocalDate.now()): Boolean {
    val eventDate = runCatching { LocalDate.parse(date) }.getOrNull() ?: return false
    val days = ChronoUnit.DAYS.between(eventDate, today)
    return days in 0L..3L
}
