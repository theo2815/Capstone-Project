package com.quickpitik.mobile.ui.runner

import com.quickpitik.mobile.data.remote.EventDto
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit

// Operational lifecycle mirrors the website's event-catalog.ts, using the event
// date rather than stored status. Discovery uses deriveEventDisplayState below
// for its race-day-only labels without changing syncing or gallery sockets.
//
// Day math (anchored to local midnight):
//   days = today - eventDate  →  event today = 0, future < 0, past > 0
//   upcoming  days < 0
//   live      0..3        (upload window, including post-race grace)
//   open      4..89       (recent, photos all in, still searchable)
//   past      >= 90       (archive)
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

// Race-day badges and discovery filters are presentation only. Gallery sockets
// and photographer syncing keep using deriveEventState and its upload window.
fun deriveEventDisplayState(date: String, today: LocalDate = LocalDate.now()): EventState {
    val state = deriveEventState(date, today)
    return if (state == EventState.LIVE && date != today.toString()) EventState.OPEN else state
}

// The web extracts the city from the tail of "Venue, City" via lastIndexOf(",").
fun extractCity(location: String): String {
    val idx = location.lastIndexOf(',')
    return if (idx == -1) location.trim() else location.substring(idx + 1).trim()
}

private val DATE_LABEL_FORMATTER = DateTimeFormatter.ofPattern("MMM d, yyyy")

fun eventDateLabel(date: String): String =
    runCatching { LocalDate.parse(date).format(DATE_LABEL_FORMATTER).uppercase() }.getOrDefault(date)

// Display stage paired with the event for discovery filtering and badges.
fun EventDto.displayState(today: LocalDate = LocalDate.now()): EventState = deriveEventDisplayState(date, today)

// Photographer upload window: race-day + 3-day grace window (days 0..3).
// Backend enforces the same window server-side as EVENT_NOT_UPLOADABLE.
fun canUploadToEvent(date: String, today: LocalDate = LocalDate.now()): Boolean {
    val eventDate = runCatching { LocalDate.parse(date) }.getOrNull() ?: return false
    val days = ChronoUnit.DAYS.between(eventDate, today)
    return days in 0L..3L
}
