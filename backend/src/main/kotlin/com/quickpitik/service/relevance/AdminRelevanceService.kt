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
import com.quickpitik.dto.admin.RelevanceConsentWithdrawalDto
import com.quickpitik.dto.admin.RelevanceHiddenPhotoDto
import com.quickpitik.dto.admin.RelevancePhotoRestoreDto
import com.quickpitik.dto.admin.RelevanceRestoreDto
import com.quickpitik.dto.admin.RelevanceSampleDto
import com.quickpitik.dto.admin.RelevanceWordDto
import com.quickpitik.dto.admin.toDto
import com.quickpitik.entity.CaseAction
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.CaseVerdict
import com.quickpitik.config.RelevanceMode
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.entity.Role
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import org.springframework.http.HttpStatus
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import com.quickpitik.repository.SamplePhotoRow
import com.quickpitik.repository.UserRepository
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
    private val switch: RelevanceSwitch,
    private val strikes: RelevanceStrikeRepository,
    private val payoutHold: RelevanceHoldService,
    private val users: UserRepository,
) {

    // RV-L7 part a (legal 1.7): the photographer withdrew the consent to automatic actions, in any mode. From then on the
    // consent rule at apply and the automatic hide, scan and payout hold steps leave them out; nothing already hidden or
    // held changes. A second call keeps the first date and logs nothing.
    @Transactional
    fun withdrawConsent(adminId: UUID, photographerId: UUID): RelevanceConsentWithdrawalDto {
        requirePhotographer(photographerId)
        if (users.markRelevanceConsentWithdrawn(photographerId, clock.postgresTimestamp()) == 1) {
            decisionLog.logUserDecision(adminId, photographerId, "relevance_consent_withdrawn")
        }
        return consentWithdrawal(photographerId)
    }

    @Transactional
    fun clearConsentWithdrawal(adminId: UUID, photographerId: UUID): RelevanceConsentWithdrawalDto {
        requirePhotographer(photographerId)
        if (users.clearRelevanceConsentWithdrawn(photographerId) == 1) {
            decisionLog.logUserDecision(adminId, photographerId, "relevance_consent_withdrawal_cleared")
        }
        return consentWithdrawal(photographerId)
    }

    private fun requirePhotographer(id: UUID) {
        if (users.findById(id).orElse(null)?.role != Role.PHOTOGRAPHER) {
            throw NotFoundException(code = ErrorCodes.USER_NOT_FOUND, message = "Photographer not found")
        }
    }

    private fun consentWithdrawal(id: UUID) = RelevanceConsentWithdrawalDto(id, users.relevanceConsentWithdrawnAt(id)?.toString())

    // RV-2.5b (G10): the admin closes a current Can't tell case LOOKS_FINE, in any mode. The case is then the batch's
    // comparator (P21), so the next Can't tell is closed SAME_VERDICT. A case that hides photos (an admin hide) is
    // refused: the admin restores it instead, so no admin hold is left with no case acting on it.
    @Transactional
    fun looksFine(adminId: UUID, caseId: UUID, note: String?): RelevanceCaseRowDto {
        val case = load(caseId)
        if (case.stage != CaseStage.APPLIED.name || case.verdict != CaseVerdict.CANT_TELL.name || case.action != CaseAction.NONE.name) {
            throw wrongState()
        }
        if (cases.closeLooksFine(case.id, adminId, clock.postgresTimestamp()) != 1) throw wrongState()
        logDecision(adminId, case, "relevance_looks_fine", note)
        return cases.caseRow(caseId)!!.toDto()
    }

    // RV-2.5b (G10): the admin hides a current Can't tell batch, in auto-hide only. The route changes only the case
    // (HIDE_ALL, verdict_note ADMIN_HIDE; the verdict stays Can't tell, so it never counts) and logs the decision; the
    // hold step places ADMIN holds a chunk at a time. A case applied in record-only never acts (P18), so it is refused.
    // RV-4.5a: except a no-consent case whose verdict would have hidden the whole batch; ADMIN_HIDE keeps it uncounted.
    @Transactional
    fun hideBatch(adminId: UUID, caseId: UUID, note: String?): RelevanceCaseRowDto {
        val case = load(caseId)
        if (switch.mode != RelevanceMode.AUTO_HIDE) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.RELEVANCE_ACTIONS_OFF, "Hiding a batch needs the relevance mode auto-hide")
        }
        if (cases.markAdminHide(case.id) != 1) throw wrongState()
        logDecision(adminId, case, "relevance_admin_hidden", note)
        return cases.caseRow(caseId)!!.toDto()
    }

    // RV-4.5a: the admin holds the payouts of a current no-consent Copied case that holds none yet, in auto-hide only (the
    // same mode rule as hide-batch). The case is locked first, as End hold does.
    @Transactional
    fun holdPayouts(adminId: UUID, caseId: UUID, note: String?): RelevanceCaseRowDto {
        cases.lockCase(caseId)
        val case = load(caseId)
        if (switch.mode != RelevanceMode.AUTO_HIDE) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.RELEVANCE_ACTIONS_OFF, "Holding payouts needs the relevance mode auto-hide")
        }
        if (!payoutHold.placeByAdmin(case, adminId)) throw wrongState()
        logDecision(adminId, case, "relevance_payouts_held", note)
        return cases.caseRow(caseId)!!.toDto()
    }

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
                    // RV-4.2: a copied case's payout hold ends with it (the case row is held by the close above).
                    payoutHold.end(case.id, adminId, restore = true)
                }
                // RVF-11: any closed case of a batch with no current case, so the junk a won appeal left held can be
                // released. A repeat click of an admin restore stays unlogged.
                isClosedWithNoCurrent(case) -> if (case.closeReason != RESTORED_BY_ADMIN) logDecision(adminId, case, "relevance_restored", note)
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

    // RV-4.2 (A.8): "End hold". The case is locked first, so the order is the case, then the photographer's user row and the
    // flag (AdminFlagService.loadFlag). The flag is resolved, which sends the existing "Payout hold lifted" note; 409 when
    // the case holds no payouts.
    @Transactional
    fun endHold(adminId: UUID, caseId: UUID): RelevanceCaseRowDto {
        val case = load(caseId)
        cases.lockCase(case.id)
        if (!payoutHold.end(case.id, adminId, restore = false)) throw wrongState()
        logDecision(adminId, case, "relevance_hold_ended", null)
        return cases.caseRow(caseId)!!.toDto()
    }

    // RV-2.5a: one photo of the current case's batch (RVF-11: or of a closed case's batch with no current case) comes back
    // by the one restore rule; another hold or a hidden photo flag may still keep it, and the answer says so. Its rows take
    // the case, so this case never hides it again.
    @Transactional
    fun restorePhoto(adminId: UUID, caseId: UUID, photoId: UUID): RelevancePhotoRestoreDto {
        val case = load(caseId)
        if (case.stage != CaseStage.APPLIED.name && !isClosedWithNoCurrent(case)) throw wrongState()
        val batch = batches.findById(case.batchId).orElseThrow()
        holds.photoStatusInBatch(photoId, batch.eventId, batch.photographerId)
            ?: throw NotFoundException("Photo not found in this batch", ErrorCodes.PHOTO_NOT_FOUND)
        // Logged only when a hold was stamped: a second click, or a photo only another kind keeps, restored nothing.
        if (unhide(holds.restorePhotoByAdmin(case.batchId, photoId, case.id, clock.postgresTimestamp())) > 0) {
            logDecision(adminId, case, "relevance_photo_restored", null, photoId)
        }
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

    private fun isClosedWithNoCurrent(case: RelevanceCase) = case.stage == CaseStage.CLOSED.name && cases.currentFor(case.batchId) == null

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

    fun listCases(tab: String, page: PaginationParams, repeat: Boolean = false): PaginatedResponse<RelevanceCaseRowDto> {
        if (tab !in TABS) throw ValidationException("tab must be one of ${TABS.joinToString()}", field = "tab")
        val result = cases.pageCurrentCases(tab, repeat, strikesSince(), OffsetLimitPageable(page))
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
            confirmedAt = c.confirmedAt?.toInstant()?.toString(),
            photographerStrikes = strikes.countSince(batches.findById(row.batchId).orElseThrow().photographerId, strikesSince()),
            payoutHoldAt = cases.liveHoldAt(caseId)?.toString(),
            closeReason = c.closeReason,
            payoutHoldPlaced = c.holdFlagId != null,
        )
    }

    // The start of the ladder's rolling window (A.7, G6).
    private fun strikesSince() = clock.postgresTimestamp().minusMonths(props.strikes.windowMonths.toLong())

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
