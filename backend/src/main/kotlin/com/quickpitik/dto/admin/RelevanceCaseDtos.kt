package com.quickpitik.dto.admin

import com.quickpitik.entity.CaseAction
import com.quickpitik.repository.RelevanceAlertRow
import com.quickpitik.repository.RelevanceCaseRow
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
