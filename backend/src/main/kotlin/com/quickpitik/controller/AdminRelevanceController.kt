package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.RelevanceActionRequest
import com.quickpitik.dto.admin.RelevanceAlertDto
import com.quickpitik.dto.admin.RelevanceCaseDetailDto
import com.quickpitik.dto.admin.RelevanceCaseRowDto
import com.quickpitik.dto.admin.RelevanceHiddenPhotoDto
import com.quickpitik.dto.admin.RelevancePhotoRestoreDto
import com.quickpitik.dto.admin.RelevanceRestoreDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.relevance.AdminRelevanceService
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// RV-1.10 (plan 5.11): the admin Photo Relevance page. Read routes plus marking an alert seen; RV-2.5a adds the
// restore and keep-hidden actions and the hidden-photos read.
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

    // RV-2.5a (G10): the admin's actions on a case. Each logs the admin's decision.
    @PostMapping("/cases/{caseId}/restore")
    fun restore(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @Valid @RequestBody(required = false) body: RelevanceActionRequest?,
    ): RelevanceRestoreDto = adminRelevanceService.restore(principal.userId, caseId, body?.note)

    @PostMapping("/cases/{caseId}/keep-hidden")
    fun keepHidden(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @Valid @RequestBody(required = false) body: RelevanceActionRequest?,
    ): RelevanceCaseRowDto = adminRelevanceService.keepHidden(principal.userId, caseId, body?.note)

    @PostMapping("/cases/{caseId}/photos/{photoId}/restore")
    fun restorePhoto(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @PathVariable photoId: UUID,
    ): RelevancePhotoRestoreDto = adminRelevanceService.restorePhoto(principal.userId, caseId, photoId)

    @GetMapping("/cases/{caseId}/hidden-photos")
    fun hiddenPhotos(
        @PathVariable caseId: UUID,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<RelevanceHiddenPhotoDto> = adminRelevanceService.hiddenPhotos(caseId, PaginationParams.of(offset, limit))

    @PostMapping("/alerts/{alertId}/seen")
    fun markSeen(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable alertId: UUID,
    ): RelevanceAlertDto = adminRelevanceService.markSeen(principal.userId, alertId)
}
