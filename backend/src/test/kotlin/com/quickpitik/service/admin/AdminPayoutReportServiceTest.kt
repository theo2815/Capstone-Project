package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.admin.AcknowledgePayoutReportRequest
import com.quickpitik.dto.admin.ResolvePayoutReportRequest
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.util.UUID
import kotlin.test.*

class AdminPayoutReportServiceTest {
    private val reports = Mockito.mock(PayoutReportRepository::class.java)
    private val decisions = Mockito.mock(AdminDecisionLogService::class.java)
    private val service = AdminPayoutReportService(reports, Mockito.mock(UserRepository::class.java),
        Mockito.mock(PhotographerSettingsRepository::class.java), decisions)
    private val actor = UUID.randomUUID()

    private fun report(status: PayoutReportStatus): PayoutReport {
        val report = PayoutReport(payoutId = "test-cycle", photographerId = UUID.randomUUID(),
            reasonWire = "other", statusWire = status.wire)
        Mockito.`when`(reports.findByIdForUpdate(report.id)).thenReturn(report)
        Mockito.`when`(decisions.logPayoutDecision(actor, report.payoutId, "report_acknowledged", null,
            mapOf("reportId" to report.id.toString(), "reply" to "Received")))
            .thenReturn(AdminDecisionLog(adminId = actor, decision = "report_acknowledged"))
        Mockito.`when`(decisions.logPayoutDecision(actor, report.payoutId, "report_resolved", null,
            mapOf("reportId" to report.id.toString(), "resolutionNote" to "Done")))
            .thenReturn(AdminDecisionLog(adminId = actor, decision = "report_resolved"))
        return report
    }

    @Test fun `acknowledge accepts only open reports and rejects a second reply`() {
        val report = report(PayoutReportStatus.OPEN)
        assertEquals("acknowledged", service.acknowledge(actor, report.id, AcknowledgePayoutReportRequest("Received")).status)
        val error = assertFailsWith<ApiException> {
            service.acknowledge(actor, report.id, AcknowledgePayoutReportRequest("Different"))
        }
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, error.code)
        assertEquals("Received", report.acknowledgeReply)
    }

    @Test fun `resolve accepts open or acknowledged but preserves the first resolution`() {
        for (status in listOf(PayoutReportStatus.OPEN, PayoutReportStatus.ACKNOWLEDGED)) {
            val report = report(status)
            assertEquals("resolved", service.resolve(actor, report.id, ResolvePayoutReportRequest("Done")).status)
            assertFailsWith<ApiException> { service.resolve(actor, report.id, ResolvePayoutReportRequest("Different")) }
            assertFailsWith<ApiException> { service.acknowledge(actor, report.id, AcknowledgePayoutReportRequest("Received")) }
            assertEquals("Done", report.resolutionNote)
        }
    }
}
