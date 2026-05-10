package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AcknowledgePayoutReportRequest
import com.quickpitik.dto.admin.AdminPayoutReportDto
import com.quickpitik.dto.admin.ResolvePayoutReportRequest
import com.quickpitik.entity.PayoutReport
import com.quickpitik.entity.PayoutReportStatus
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutReportRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class AdminPayoutReportService(
    private val payoutReportRepository: PayoutReportRepository,
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val adminDecisionLogService: AdminDecisionLogService,
) {

    @Transactional(readOnly = true)
    fun list(
        statusFilter: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminPayoutReportDto> {
        val statusWire = statusFilter?.takeIf { it.isNotBlank() }?.let { parseStatus(it).wire }
        val page = payoutReportRepository.pageForAdmin(statusWire, OffsetLimitPageable(params))
        if (page.isEmpty) return PaginatedResponse.empty(params)
        val items = page.content.map { hydrateOne(it) }
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    fun acknowledge(
        adminId: UUID,
        reportId: UUID,
        req: AcknowledgePayoutReportRequest,
    ): AdminPayoutReportDto {
        val report = loadReport(reportId)
        if (report.status != PayoutReportStatus.OPEN) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Report is already ${report.status.wire}",
            )
        }
        report.status = PayoutReportStatus.ACKNOWLEDGED
        report.acknowledgedAt = OffsetDateTime.now()
        report.acknowledgeReply = req.reply.trim()
        payoutReportRepository.save(report)

        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = report.payoutId,
            decision = "report_acknowledged",
            meta = mapOf("reportId" to report.id.toString(), "reply" to req.reply),
        )
        adminDecisionLogService.pushMessage(
            photographerId = report.photographerId,
            kind = PhotographerMessageKind.PAYOUT_REPORT_ACKNOWLEDGED,
            body = "Your report on cycle ${report.payoutId} was acknowledged: ${req.reply}",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(report)
    }

    fun resolve(
        adminId: UUID,
        reportId: UUID,
        req: ResolvePayoutReportRequest,
    ): AdminPayoutReportDto {
        val report = loadReport(reportId)
        if (report.status == PayoutReportStatus.RESOLVED) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Report is already resolved",
            )
        }
        report.status = PayoutReportStatus.RESOLVED
        report.resolvedAt = OffsetDateTime.now()
        report.resolutionNote = req.resolutionNote.trim()
        payoutReportRepository.save(report)

        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = report.payoutId,
            decision = "report_resolved",
            meta = mapOf("reportId" to report.id.toString(), "resolutionNote" to req.resolutionNote),
        )
        adminDecisionLogService.pushMessage(
            photographerId = report.photographerId,
            kind = PhotographerMessageKind.PAYOUT_REPORT_RESOLVED,
            body = "Your report on cycle ${report.payoutId} was resolved. ${req.resolutionNote}",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(report)
    }

    private fun loadReport(reportId: UUID): PayoutReport =
        payoutReportRepository.findById(reportId).orElseThrow {
            NotFoundException(code = ErrorCodes.REPORT_NOT_FOUND, message = "Report not found")
        }

    private fun parseStatus(raw: String): PayoutReportStatus =
        runCatching { PayoutReportStatus.fromWire(raw) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "status must be one of open / acknowledged / resolved",
                field = "status",
            )
        }

    internal fun hydrateOne(report: PayoutReport): AdminPayoutReportDto {
        val user = userRepository.findById(report.photographerId).orElse(null)
        val settings = photographerSettingsRepository.findById(report.photographerId).orElse(null)
        return AdminPayoutReportDto(
            id = report.id,
            payoutCycleId = report.payoutId,
            photographerId = report.photographerId,
            photographerName = user?.name ?: "—",
            handle = settings?.handle,
            reason = report.reasonWire,
            note = report.note,
            status = report.status.wire,
            reportedAt = report.openedAt.toString(),
            acknowledgedAt = report.acknowledgedAt?.toString(),
            acknowledgeReply = report.acknowledgeReply,
            resolvedAt = report.resolvedAt?.toString(),
            resolutionNote = report.resolutionNote,
        )
    }
}
