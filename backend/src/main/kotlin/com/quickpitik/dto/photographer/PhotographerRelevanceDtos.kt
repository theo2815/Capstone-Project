package com.quickpitik.dto.photographer

import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

// RV-3.2a: GET /api/v1/me/photographer/relevance/cases/{caseId}. What a photographer may see of their own Photo
// Relevance case: the verdict and its numbers, never the model's reason, the summary sent or the words read from photos
// (G12). appealDeadline is the last day to appeal (Philippine date), null when the case has no appeal clock.
// RV-3.4a: eventDate is the event's date (V2-T13 names it); closeReason says why a closed case closed (null while open),
// so the page can show the note that close sent.
data class PhotographerRelevanceCaseDto(
    val caseId: UUID,
    val eventId: UUID,
    val eventName: String,
    val verdict: String?,
    val action: String?,
    val current: Boolean,
    val batchPhotos: Int,
    val sampleSize: Int?,
    val junkCount: Int?,
    val hiddenPhotos: Int,
    val scanMeasured: Int?,
    val scanBlurry: Int?,
    val noticeAt: Instant?,
    val appealDeadline: LocalDate?,
    val appeal: PhotographerAppealDto?,
    val eventDate: LocalDate? = null,
    val closeReason: String? = null,
    // RV-4.5a: the page shows the inbox notes' sentences by the same rules. payoutHoldBinds: a payout hold binds the case
    // right now ("your payouts stay on hold"); lostAppealCounts: a lost appeal would count as a confirmed case (false for
    // an admin's hide, which never counts).
    val payoutHoldBinds: Boolean = false,
    val lostAppealCounts: Boolean = true,
)

// The appeal's state only: the admin's note is never shown (V2-T19), and the photographer's own reason is not echoed.
data class PhotographerAppealDto(
    val id: UUID,
    val status: String,
    val createdAt: OffsetDateTime?,
    val decidedAt: OffsetDateTime?,
)

// POST /api/v1/me/photographer/relevance/cases/{caseId}/appeal. The reason is 1 to 1,000 characters once trimmed.
data class FileAppealRequest(val reason: String = "")

// RV-3.3: GET /api/v1/me/photographer/relevance/cases/{caseId}/blurry. One measured blurry photo of the batch that is
// still in the gallery; thumbnailUrl is a presigned preview (the watermarked one, else the thumbnail), never the original.
data class PhotographerBlurryPhotoDto(val photoId: UUID, val thumbnailUrl: String?)
