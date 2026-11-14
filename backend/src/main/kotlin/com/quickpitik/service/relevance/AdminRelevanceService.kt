package com.quickpitik.service.relevance

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.RelevanceAlertDto
import com.quickpitik.dto.admin.RelevanceCaseDetailDto
import com.quickpitik.dto.admin.RelevanceCaseRowDto
import com.quickpitik.dto.admin.RelevanceSampleDto
import com.quickpitik.dto.admin.RelevanceWordDto
import com.quickpitik.dto.admin.toDto
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.SamplePhotoRow
import com.quickpitik.service.events.postgresTimestamp
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.Clock
import java.util.UUID

// RV-1.10 (plan 5.11): the reads behind the admin Photo Relevance page, and marking an alert seen. It reads cases,
// not flags, so it works in record-only with the admin flags switch off (no AdminFlagService.ensureEnabled).
@Service
@Transactional(readOnly = true)
class AdminRelevanceService(
    private val cases: RelevanceCaseRepository,
    private val alerts: RelevanceAlertRepository,
    private val batches: BatchRelevanceRepository,
    private val props: RelevanceProperties,
    private val storage: StorageService,
    private val storageProperties: StorageProperties,
    private val clock: Clock,
) {

    fun listCases(tab: String, page: PaginationParams): PaginatedResponse<RelevanceCaseRowDto> {
        if (tab !in TABS) throw ValidationException("tab must be one of ${TABS.joinToString()}", field = "tab")
        // Appeals arrive with part 2; until then the tab is empty.
        if (tab == "appeals") return PaginatedResponse.empty(page)
        val result = cases.pageCurrentCases(tab, OffsetLimitPageable(page))
        return PaginatedResponse.of(result.content.map { it.toDto() }, result.totalElements, page)
    }

    fun case(caseId: UUID): RelevanceCaseDetailDto {
        val row = cases.caseRow(caseId) ?: throw NotFoundException("Relevance case not found", ErrorCodes.RELEVANCE_CASE_NOT_FOUND)
        val c = cases.findById(caseId).orElseThrow()
        val (junk, clean) = cases.measuredSample(caseId).partition { isJunk(it.scores(), props) == true }
        return RelevanceCaseDetailDto(
            case = row.toDto(),
            stage = c.stage,
            sampleSize = c.sampleSize,
            measured = c.measured,
            blurry = c.blurry,
            dark = c.dark,
            notRace = c.notRace,
            junk = c.junk,
            band = c.band,
            modelVerdict = c.modelVerdict,
            adminReason = c.adminReason,
            evidenceUsed = c.evidenceUsed?.split(',')?.filter { it.isNotBlank() } ?: emptyList(),
            summarySent = c.summarySent,
            model = c.model,
            promptVersion = c.promptVersion,
            wordsPhotos = c.wordsPhotos,
            topWords = batches.topWords(row.batchId, props.words.minSharePercent, props.words.topN)
                .map { RelevanceWordDto(it.word, it.photos) },
            junkSamples = junk.take(SAMPLES).map { it.toSample() },
            cleanSamples = clean.take(SAMPLES).map { it.toSample() },
            alerts = alerts.alertsForBatch(row.batchId).map { it.toDto() },
        )
    }

    fun listAlerts(unseenOnly: Boolean, page: PaginationParams): PaginatedResponse<RelevanceAlertDto> {
        val result = alerts.pageAlerts(unseenOnly, OffsetLimitPageable(page))
        return PaginatedResponse.of(result.content.map { it.toDto() }, result.totalElements, page)
    }

    // The one write. Marking an alert seen twice keeps the first admin and time.
    @Transactional
    fun markSeen(adminId: UUID, alertId: UUID): RelevanceAlertDto {
        alerts.markSeen(alertId, adminId, clock.postgresTimestamp())
        return alerts.alertRow(alertId)?.toDto()
            ?: throw NotFoundException("Relevance alert not found", ErrorCodes.RELEVANCE_ALERT_NOT_FOUND)
    }

    private fun SamplePhotoRow.scores() =
        PhotoScores(fgSharpness, fgBrightness, sharpness, brightness, personConf, raceConf, measureError = null)

    private fun SamplePhotoRow.toSample(): RelevanceSampleDto {
        val s = scores()
        // The watermarked preview, else the thumbnail; never the original.
        val key = watermarkKey ?: thumbnailKey
        return RelevanceSampleDto(
            photoId = photoId,
            thumbnailUrl = key?.let { storage.presignedGetUrl(it, storageProperties.presignedTtl.thumbnail) },
            fgSharpness = fgSharpness,
            fgBrightness = fgBrightness,
            sharpness = sharpness,
            brightness = brightness,
            blurry = isBlurry(s, props),
            dark = isDark(s, props),
            notRace = isNotRace(s, props),
        )
    }

    private companion object {
        val TABS = listOf("hidden", "appeals", "cant_tell", "all")
        // As many as the flags page shows (AdminFlagService EVIDENCE_SAMPLES).
        const val SAMPLES = 12
    }
}
