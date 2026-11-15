package com.quickpitik.dto.admin

import com.quickpitik.entity.CaseAction
import com.quickpitik.repository.RelevanceAlertRow
import com.quickpitik.repository.RelevanceCaseRow
import jakarta.validation.constraints.Size
import java.util.UUID

// RV-1.10 (plan 5.11): the admin Photo Relevance page. Admins only: Claude's reason and the summary sent are never
// shown to a photographer (G12). wouldHave = applied in record-only, so the action was recorded, not taken.
data class RelevanceCaseRowDto(
    val caseId: UUID,
    val eventId: UUID,
    val eventName: String,
    val photographerHandle: String?,
    val photographerName: String,
    val verdict: String?,
    val verdictNote: String?,
    val action: String?,
    val appliedMode: String?,
    val wouldHave: Boolean,
    val batchPhotos: Int,
    // Junk share of the measured sample, rounded down; null while nothing was measured.
    val junkPercent: Int?,
    val decidedAt: String?,
)

data class RelevanceCaseDetailDto(
    val case: RelevanceCaseRowDto,
    val stage: String,
    val sampleSize: Int?,
    val measured: Int?,
    val blurry: Int?,
    val dark: Int?,
    val notRace: Int?,
    val junk: Int?,
    val band: String?,
    val modelVerdict: String?,
    val adminReason: String?,
    val evidenceUsed: List<String>,
    val summarySent: String?,
    val model: String?,
    val promptVersion: String?,
    val wordsPhotos: Int?,
    val topWords: List<RelevanceWordDto>,
    val junkSamples: List<RelevanceSampleDto>,
    val cleanSamples: List<RelevanceSampleDto>,
    val alerts: List<RelevanceAlertDto>,
    // RV-4.1 (A.7): when the case was confirmed (null: not confirmed), and the batch photographer's confirmed cases in
    // the last strikes.window-months months, across events.
    val confirmedAt: String? = null,
    val photographerStrikes: Int = 0,
    // RV-4.2 (A.8): when the case's automatic payout hold started, while it binds (null: no live hold); then the case's
    // close reason (null while it is current), so the page can tell why a closed case closed.
    val payoutHoldAt: String? = null,
    val closeReason: String? = null,
)

data class RelevanceWordDto(val word: String, val photos: Int)

// One measured sample photo with its stored scores (0-100) and the junk rules' answers; null = unknown (a cut-off
// is unset). thumbnailUrl is a presigned preview, never the original (G19); null while no preview exists.
data class RelevanceSampleDto(
    val photoId: UUID,
    val thumbnailUrl: String?,
    val fgSharpness: Double?,
    val fgBrightness: Double?,
    val sharpness: Double?,
    val brightness: Double?,
    val blurry: Boolean?,
    val dark: Boolean?,
    val notRace: Boolean?,
)

data class RelevanceAlertDto(
    val alertId: UUID,
    val caseId: UUID?,
    val eventId: UUID,
    val eventName: String,
    val photographerHandle: String?,
    val photographerName: String,
    val kind: String,
    val wouldHave: Boolean,
    val photos: Int?,
    val createdAt: String?,
    val seenAt: String?,
)

fun RelevanceCaseRow.toDto() = RelevanceCaseRowDto(
    caseId = caseId,
    eventId = eventId,
    eventName = eventName,
    photographerHandle = photographerHandle,
    photographerName = photographerName,
    verdict = verdict,
    verdictNote = verdictNote,
    action = action,
    appliedMode = appliedMode,
    wouldHave = appliedMode == "RECORD_ONLY" && action != null && action != CaseAction.NONE.name,
    batchPhotos = batchPhotos,
    junkPercent = measured?.takeIf { it > 0 }?.let { (junk ?: 0) * 100 / it },
    decidedAt = decidedAt?.toString(),
)

fun RelevanceAlertRow.toDto() = RelevanceAlertDto(
    alertId = alertId,
    caseId = caseId,
    eventId = eventId,
    eventName = eventName,
    photographerHandle = photographerHandle,
    photographerName = photographerName,
    kind = kind,
    wouldHave = wouldHave,
    photos = photos,
    createdAt = createdAt?.toString(),
    seenAt = seenAt?.toString(),
)

// RV-2.5a: the admin's note on a case action (optional; blank is stored as none).
data class RelevanceActionRequest(
    @field:Size(max = 1000)
    val note: String? = null,
)

// RV-2.5a: a batch restore; restoredPhotos counts the holds this call restored (0 on a repeat with nothing left).
data class RelevanceRestoreDto(val caseId: UUID, val restoredPhotos: Int)

// RV-2.5a: one photo restored; status is the photo's status after, HIDDEN while another hold or a hidden flag keeps it.
data class RelevancePhotoRestoreDto(val photoId: UUID, val status: String)

// RV-2.5a: one hold of the batch, restored ones included; thumbnailUrl is a presigned preview, never the original.
data class RelevanceHiddenPhotoDto(
    val photoId: UUID,
    val heldBy: String,
    val hiddenAt: String?,
    val restoredAt: String?,
    val restoredBy: String?,
    val status: String,
    val thumbnailUrl: String?,
)

// RV-3.2 part b (A.6): one appeal on the admin page. Admins only: the photographer's reason and the admin's decision note
// are shown here and never to a photographer. eventName is null only if the event row itself is gone.
data class RelevanceAppealRowDto(
    val appealId: UUID,
    val caseId: UUID,
    val batchId: UUID,
    val eventId: UUID,
    val eventName: String?,
    val photographerName: String?,
    val photographerHandle: String?,
    val verdict: String?,
    val verdictNote: String?,
    val action: String?,
    val caseStage: String,
    val status: String,
    val reason: String,
    val createdAt: java.time.Instant?,
    val evidenceOpenedAt: java.time.Instant?,
    val evidenceOpenedBy: UUID?,
    val decidedAt: java.time.Instant?,
    val decidedBy: UUID?,
    val decisionNote: String?,
)

fun com.quickpitik.repository.RelevanceAppealRow.toDto() = RelevanceAppealRowDto(appealId, caseId, batchId, eventId, eventName,
    photographerName, photographerHandle, verdict, verdictNote, action, caseStage, status, reason, createdAt, evidenceOpenedAt,
    evidenceOpenedBy, decidedAt, decidedBy, decisionNote)

// POST /api/v1/admin/relevance/appeals/{id}/decide: outcome WON or LOST; the note is kept for admins only.
data class RelevanceDecideRequest(
    val outcome: String = "",
    @field:Size(max = 1000)
    val note: String? = null,
)
