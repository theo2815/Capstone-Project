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
// Day math (anchored to local midnight, exactly like daysSinceEvent on the web):
//   days = today - eventDate  →  event today = 0, future < 0, past > 0
//   upcoming  days < 0
//   live      0..3   (race day + 3-day upload grace)
//   open      4..89  (recent, photos all in, still searchable)
//   past      >= 90  (archive)
enum class EventState { UPCOMING, LIVE, OPEN, PAST }

fun deriveEventState(date: String, today: LocalDate = LocalDate.now()): EventState {
    val eventDate = runCatching { LocalDate.parse(date) }.getOrNull() ?: return EventState.OPEN
    val days = ChronoUnit.DAYS.between(eventDate, today)
    return when {
        days < 0L -> EventState.UPCOMING
        days <= 3L -> EventState.LIVE
        days <= 89L -> EventState.OPEN
        else -> EventState.PAST
    }
}

// The web extracts the city from the tail of "Venue, City" via lastIndexOf(",").
fun extractCity(location: String): String {
    val idx = location.lastIndexOf(',')
    return if (idx == -1) location.trim() else location.substring(idx + 1).trim()
}

private val DATE_LABEL_FORMATTER = DateTimeFormatter.ofPattern("MMM d, yyyy")

fun eventDateLabel(date: String): String =
    runCatching { LocalDate.parse(date).format(DATE_LABEL_FORMATTER).uppercase() }.getOrDefault(date)

// Lifecycle stage paired with the event, computed once so filtering/segmenting reuses it.
fun EventDto.lifecycleState(today: LocalDate = LocalDate.now()): EventState = deriveEventState(date, today)
