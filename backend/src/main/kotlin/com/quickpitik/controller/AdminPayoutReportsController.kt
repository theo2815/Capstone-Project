package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AcknowledgePayoutReportRequest
import com.quickpitik.dto.admin.AdminPayoutReportDto
import com.quickpitik.dto.admin.ResolvePayoutReportRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminPayoutReportService
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

@RestController
@RequestMapping("/api/v1/admin/payouts/reports")
@PreAuthorize("hasRole('ADMIN')")
class AdminPayoutReportsController(
    private val adminPayoutReportService: AdminPayoutReportService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminPayoutReportDto> =
        adminPayoutReportService.list(status, PaginationParams.of(offset, limit))

    @PatchMapping("/{reportId}/acknowledge")
    fun acknowledge(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable reportId: UUID,
        @Valid @RequestBody body: AcknowledgePayoutReportRequest,
    ): AdminPayoutReportDto = adminPayoutReportService.acknowledge(principal.userId, reportId, body)

    @PatchMapping("/{reportId}/resolve")
    fun resolve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable reportId: UUID,
        @Valid @RequestBody body: ResolvePayoutReportRequest,
    ): AdminPayoutReportDto = adminPayoutReportService.resolve(principal.userId, reportId, body)
}
