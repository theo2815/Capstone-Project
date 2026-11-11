package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceProperties
import com.quickpitik.service.events.EventDtoMapper
import java.time.DateTimeException
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset

// PE-1 relevance rules (D20, D21), pure so they are tested without a database.
// Version 1 computes only Low; missing evidence is neutral.

// D20: Low when hits >= minHits and hits are at least minHitSharePercent of the look-ups.
internal fun copyLow(tightHits: Int, lookups: Int, p: RelevanceProperties.Copy): Boolean =
    tightHits >= p.minHits && tightHits * 100 >= p.minHitSharePercent * lookups

// D21: the capture instant shown in Manila; with no usable offset, the camera's own date.
internal fun manilaDate(takenAtLocal: LocalDateTime, takenAtOffset: String?): LocalDate {
    val offset = takenAtOffset?.takeIf { it.isNotBlank() }?.let {
        try { ZoneOffset.of(it) } catch (_: DateTimeException) { null }
    } ?: return takenAtLocal.toLocalDate()
    return takenAtLocal.atOffset(offset).atZoneSameInstant(EventDtoMapper.PH_ZONE).toLocalDate()
}

internal fun outsideWindow(captured: LocalDate, eventDate: LocalDate, windowDays: Long): Boolean =
    captured < eventDate.minusDays(windowDays) || captured > eventDate.plusDays(windowDays)

// D21: null (wait) while any metadata row is unread; never Low below minDatedPhotos.
internal fun dateLow(dated: Int, outside: Int, pending: Int, p: RelevanceProperties.Date): Boolean? = when {
    pending > 0 -> null
    dated < p.minDatedPhotos -> false
    else -> outside * 100 >= p.outsideSharePercent * dated
}

// Mirrors PhotoUploadService.gate (the upload window); change both together.
internal fun lastUploadDate(eventDate: LocalDate, sharedRace: Boolean): LocalDate =
    if (sharedRace) eventDate.plusDays(3) else eventDate.plusMonths(1)

internal data class UploadCutoffs(val adminClosedOnOrBefore: LocalDate, val ownerClosedOnOrBefore: LocalDate)

// Event dates on or before these have closed uploads by `today` (Manila). The owner
// cutoff is up to three days late, never early, because month lengths differ.
internal fun uploadCutoffs(today: LocalDate): UploadCutoffs = UploadCutoffs(
    adminClosedOnOrBefore = today.minusDays(4),
    ownerClosedOnOrBefore = today.minusMonths(1).minusDays(3),
)
