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
import com.quickpitik.dto.admin.RelevanceHiddenPhotoDto
import com.quickpitik.dto.admin.RelevancePhotoRestoreDto
import com.quickpitik.dto.admin.RelevanceRestoreDto
import com.quickpitik.dto.admin.RelevanceSampleDto
import com.quickpitik.dto.admin.RelevanceWordDto
import com.quickpitik.dto.admin.toDto
import com.quickpitik.entity.CaseAction
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import org.springframework.http.HttpStatus
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.repository.SamplePhotoRow
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.events.postgresTimestamp
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
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
    private val holds: RelevanceHoldRepository,
    private val photos: PhotoRepository,
    private val decisionLog: AdminDecisionLogService,
    private val transactions: TransactionTemplate,
) {

    // RV-2.5a (G10): the admin restores the batch. The current case is closed RESTORED_BY_ADMIN (it never counts and stays
    // the batch's comparator, P21) with the admin's decision; then every live automatic and admin hold of the batch is
    // restored in this request, a chunk per short transaction, in any mode (the tick returns first while the mode is
    // off). Photographer holds and flags are never touched. Calling it again on the restored case finishes what was left
    // and logs nothing more, as long as no newer case is current.
    // shortcut: a very large batch restores inside one request (chunk-size holds per transaction, so a dropped connection
    // loses no finished chunk); a second click finishes it, and in auto-hide the tick's sweep does. Move it to a job
    // if such requests start to time out.
    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun restore(adminId: UUID, caseId: UUID, note: String?): RelevanceRestoreDto {
        val now = clock.postgresTimestamp()
        val batchId = transactions.execute {
            val case = load(caseId)
            when {
                case.stage == CaseStage.APPLIED.name -> {
                    if (cases.closeByAdmin(case.id, RESTORED_BY_ADMIN, adminId, now) != 1) throw wrongState()
                    logDecision(adminId, case, "relevance_restored", note)
                }
                case.stage == CaseStage.CLOSED.name && case.closeReason == RESTORED_BY_ADMIN && cases.currentFor(case.batchId) == null -> Unit
                else -> throw wrongState()
            }
            case.batchId
        }!!
        var restored = 0
        while (true) {
            // A newer case that became current meanwhile owns the holds from now on.
            val n = transactions.execute {
                if (cases.currentFor(batchId) != null) 0 else unhide(holds.restoreByAdminChunk(batchId, now, props.hide.chunkSize))
            }!!
            if (n == 0) break
            restored += n
        }
        return RelevanceRestoreDto(caseId, restored)
    }

    // RV-2.5a: the admin keeps a hiding case as it is: its unseen alerts are marked seen and the decision logged. Nothing
    // else changes; the case is not confirmed early (that is RV-4.1's job).
    @Transactional
    fun keepHidden(adminId: UUID, caseId: UUID, note: String?): RelevanceCaseRowDto {
        val case = load(caseId)
        if (case.stage != CaseStage.APPLIED.name || case.action == null || case.action == CaseAction.NONE.name) throw wrongState()
        alerts.markCaseSeen(case.id, adminId, clock.postgresTimestamp())
        logDecision(adminId, case, "relevance_kept_hidden", note)
        return cases.caseRow(caseId)!!.toDto()
    }

    // RV-2.5a: one photo of the current case's batch comes back by the one restore rule; another hold or a hidden photo
    // flag may still keep it, and the answer says so. Its rows take the case, so this case never hides it again.
    @Transactional
    fun restorePhoto(adminId: UUID, caseId: UUID, photoId: UUID): RelevancePhotoRestoreDto {
        val case = load(caseId)
        if (case.stage != CaseStage.APPLIED.name) throw wrongState()
        val batch = batches.findById(case.batchId).orElseThrow()
        holds.photoStatusInBatch(photoId, batch.eventId, batch.photographerId)
            ?: throw NotFoundException("Photo not found in this batch", ErrorCodes.PHOTO_NOT_FOUND)
        unhide(holds.restorePhotoByAdmin(case.batchId, photoId, case.id, clock.postgresTimestamp()))
        logDecision(adminId, case, "relevance_photo_restored", null, photoId)
        return RelevancePhotoRestoreDto(photoId, holds.photoStatusInBatch(photoId, batch.eventId, batch.photographerId)!!)
    }

    // RV-2.5a: the batch's holds, restored ones included, newest first, with preview thumbnails (never the original).
    fun hiddenPhotos(caseId: UUID, page: PaginationParams): PaginatedResponse<RelevanceHiddenPhotoDto> {
        val case = load(caseId)
        val result = holds.pageBatchHolds(case.batchId, OffsetLimitPageable(page))
        return PaginatedResponse.of(result.content.map {
            RelevanceHiddenPhotoDto(it.photoId, it.heldBy, it.hiddenAt?.toString(), it.restoredAt?.toString(), it.restoredBy,
                it.status, (it.watermarkKey ?: it.thumbnailKey)?.let { key -> storage.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail) })
        }, result.totalElements, page)
    }

    private fun load(caseId: UUID): RelevanceCase = cases.findById(caseId).orElseThrow {
        NotFoundException("Relevance case not found", ErrorCodes.RELEVANCE_CASE_NOT_FOUND)
    }

    // The holds were just stamped restored: lock their photos in id order, then un-hide by the one restore rule.
    private fun unhide(ids: List<UUID>): Int {
        if (ids.isNotEmpty()) {
            photos.lockIdsForUpdate(ids)
            holds.unhideReleased(ids)
        }
        return ids.size
    }

    private fun logDecision(adminId: UUID, case: RelevanceCase, decision: String, note: String?, photoId: UUID? = null) {
        val batch = batches.findById(case.batchId).orElseThrow()
        decisionLog.logUserDecision(adminId, batch.photographerId, decision, note?.trim()?.ifBlank { null },
            meta = mapOf("caseId" to case.id.toString(), "eventId" to batch.eventId.toString()) +
                (photoId?.let { mapOf("photoId" to it.toString()) } ?: emptyMap()))
    }

    private fun wrongState() = ApiException(HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION,
        "This action does not fit the case's current state")

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
        const val RESTORED_BY_ADMIN = "RESTORED_BY_ADMIN"
        // As many as the flags page shows (AdminFlagService EVIDENCE_SAMPLES).
        const val SAMPLES = 12
    }
}
