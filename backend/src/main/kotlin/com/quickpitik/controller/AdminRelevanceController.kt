package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.RelevanceAlertDto
import com.quickpitik.dto.admin.RelevanceCaseDetailDto
import com.quickpitik.dto.admin.RelevanceCaseRowDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.relevance.AdminRelevanceService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// RV-1.10 (plan 5.11): the admin Photo Relevance page. Read routes plus marking an alert seen; the action routes
// come with part 2.
@RestController
@RequestMapping("/api/v1/admin/relevance")
@PreAuthorize("hasRole('ADMIN')")
class AdminRelevanceController(
    private val adminRelevanceService: AdminRelevanceService,
) {

    @GetMapping("/cases")
    fun cases(
        @RequestParam(required = false) tab: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<RelevanceCaseRowDto> =
        adminRelevanceService.listCases(tab ?: "all", PaginationParams.of(offset, limit))

    @GetMapping("/cases/{caseId}")
    fun case(@PathVariable caseId: UUID): RelevanceCaseDetailDto = adminRelevanceService.case(caseId)

    @GetMapping("/alerts")
    fun alerts(
        @RequestParam(required = false) unseen: Boolean?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<RelevanceAlertDto> =
        adminRelevanceService.listAlerts(unseen ?: false, PaginationParams.of(offset, limit))

    @PostMapping("/alerts/{alertId}/seen")
    fun markSeen(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable alertId: UUID,
    ): RelevanceAlertDto = adminRelevanceService.markSeen(principal.userId, alertId)
}
