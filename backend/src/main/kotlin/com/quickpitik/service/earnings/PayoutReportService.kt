package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.earnings.PayoutReportDto
import com.quickpitik.dto.earnings.SubmitPayoutReportRequest
import com.quickpitik.dto.earnings.toDto
import com.quickpitik.entity.PayoutReport
import com.quickpitik.entity.PayoutReportReason
import com.quickpitik.entity.PayoutReportStatus
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PayoutReportRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.websocket.AdminInboxEvent
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

/**
 * Photographer-scoped report submission + listing. The plan locks one OPEN
 * report per (cycle, photographer) — service does the existence check then
 * the partial unique index in V9 (`uq_payout_reports_open_per_cycle`) is the
 * second line of defense against a race that slips between the check and the
 * insert. DataIntegrityViolation is mapped back to the same 409 envelope so
 * the FE never sees a leaked DB error.
 */
@Service
@Transactional
class PayoutReportService(
    private val payoutReportRepository: PayoutReportRepository,
    private val payoutCycleRepository: PayoutCycleRepository,
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val eventPublisher: ApplicationEventPublisher,
) {
    fun submit(
        photographerId: UUID,
        cycleId: String,
        request: SubmitPayoutReportRequest,
    ): PayoutReportDto {
        val cycle = payoutCycleRepository.findByIdAndPhotographerId(cycleId, photographerId)
            ?: throw NotFoundException(code = ErrorCodes.PAYOUT_NOT_FOUND, message = "Payout not found")

        val reason = PayoutReportReason.fromWire(request.reason) ?: throw ValidationException(
            code = ErrorCodes.INVALID_REASON,
            message = "reason must be one of not_received|wrong_amount|account_info|processing_delay|other",
            field = "reason",
        )

        val existing = payoutReportRepository.findOpenForCycle(cycle.id, photographerId)
        if (existing != null) {
            throw ConflictException(
                code = ErrorCodes.REPORT_ALREADY_OPEN,
                message = "An open report already exists for this payout",
            )
        }

        val report = PayoutReport(
            payoutId = cycle.id,
            photographerId = photographerId,
            reasonWire = reason.wire,
            note = request.note?.trim().orEmpty(),
            statusWire = PayoutReportStatus.OPEN.wire,
        )
        val saved = try {
            payoutReportRepository.save(report)
        } catch (ex: DataIntegrityViolationException) {
            throw ConflictException(
                code = ErrorCodes.REPORT_ALREADY_OPEN,
                message = "An open report already exists for this payout",
            )
        }

        // Push to connected admins so the payout-reports queue updates in
        // real time. AFTER_COMMIT only — rollback discards row + push.
        eventPublisher.publishEvent(
            AdminInboxEvent(
                payload = mapOf(
                    "type" to "payout_report_filed",
                    "entityId" to saved.id.toString(),
                    "actorId" to photographerId.toString(),
                    "occurredAt" to OffsetDateTime.now().toString(),
                ),
            ),
        )

        return saved.toDto(
            photographerName = lookupName(photographerId),
            handle = lookupHandle(photographerId),
        )
    }

    @Transactional(readOnly = true)
    fun list(
        photographerId: UUID,
        cycleId: String?,
        statusWire: String?,
    ): List<PayoutReportDto> {
        // Validate the status filter early so a typo in the FE returns a clean
        // 400 envelope instead of a 500 from the entity-side fromWire throw.
        if (statusWire != null) {
            try {
                PayoutReportStatus.fromWire(statusWire)
            } catch (ex: IllegalArgumentException) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "status must be one of open|acknowledged|resolved",
                    field = "status",
                )
            }
        }

        val rows = payoutReportRepository.findForPhotographer(
            photographerId = photographerId,
            cycleId = cycleId,
            statusWire = statusWire,
        )
        if (rows.isEmpty()) return emptyList()
        val name = lookupName(photographerId)
        val handle = lookupHandle(photographerId)
        return rows.map { it.toDto(photographerName = name, handle = handle) }
    }

    private fun lookupName(photographerId: UUID): String =
        userRepository.findById(photographerId).map { it.name }.orElse("")

    private fun lookupHandle(photographerId: UUID): String? =
        photographerSettingsRepository.findById(photographerId).map { it.handle }.orElse(null)
}
