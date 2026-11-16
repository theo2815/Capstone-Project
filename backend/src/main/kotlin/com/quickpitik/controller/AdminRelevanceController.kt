package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.RelevanceActionRequest
import com.quickpitik.dto.admin.RelevanceAlertDto
import com.quickpitik.dto.admin.RelevanceCaseDetailDto
import com.quickpitik.dto.admin.RelevanceCaseRowDto
import com.quickpitik.dto.admin.RelevanceConsentWithdrawalDto
import com.quickpitik.dto.admin.RelevanceHiddenPhotoDto
import com.quickpitik.dto.admin.RelevancePhotoRestoreDto
import com.quickpitik.dto.admin.RelevanceRestoreDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.dto.admin.RelevanceAppealRowDto
import com.quickpitik.dto.admin.RelevanceDecideRequest
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.RelevanceAppealService
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.PutMapping
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
    private val appealService: RelevanceAppealService,
) {

    @GetMapping("/cases")
    fun cases(
        @RequestParam(required = false) tab: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
        // RV-4.1: only "Repeat cases" (A.7).
        @RequestParam(required = false) repeat: Boolean?,
    ): PaginatedResponse<RelevanceCaseRowDto> =
        adminRelevanceService.listCases(tab ?: "all", PaginationParams.of(offset, limit), repeat ?: false)

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

    // RV-2.5b (G10): the Can't tell actions.
    @PostMapping("/cases/{caseId}/looks-fine")
    fun looksFine(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @Valid @RequestBody(required = false) body: RelevanceActionRequest?,
    ): RelevanceCaseRowDto = adminRelevanceService.looksFine(principal.userId, caseId, body?.note)

    @PostMapping("/cases/{caseId}/hide-batch")
    fun hideBatch(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @Valid @RequestBody(required = false) body: RelevanceActionRequest?,
    ): RelevanceCaseRowDto = adminRelevanceService.hideBatch(principal.userId, caseId, body?.note)

    @PostMapping("/cases/{caseId}/keep-hidden")
    fun keepHidden(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @Valid @RequestBody(required = false) body: RelevanceActionRequest?,
    ): RelevanceCaseRowDto = adminRelevanceService.keepHidden(principal.userId, caseId, body?.note)

    // RV-4.2 (A.8): the admin ends a copied case's payout hold; the case itself stays as it is.
    @PostMapping("/cases/{caseId}/end-hold")
    fun endHold(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable caseId: UUID): RelevanceCaseRowDto =
        adminRelevanceService.endHold(principal.userId, caseId)

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

    // RV-3.2 part b (A.6): appeals. status is OPEN (the default), WON, LOST or all.
    @GetMapping("/appeals")
    fun appeals(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<RelevanceAppealRowDto> = appealService.listAppeals(status ?: "OPEN", PaginationParams.of(offset, limit))

    @PostMapping("/appeals/{appealId}/evidence-opened")
    fun evidenceOpened(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable appealId: UUID): RelevanceAppealRowDto =
        appealService.evidenceOpened(principal.userId, appealId)

    @PostMapping("/appeals/{appealId}/decide")
    fun decide(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable appealId: UUID,
        @Valid @RequestBody body: RelevanceDecideRequest,
    ): RelevanceAppealRowDto = appealService.decide(principal.userId, appealId, body.outcome, body.note)

    // RV-L7 part a (legal 1.7): the photographer withdrew the consent to automatic actions (PUT), or it is cleared (DELETE).
    @PutMapping("/photographers/{photographerId}/consent-withdrawal")
    fun withdrawConsent(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable photographerId: UUID): RelevanceConsentWithdrawalDto =
        adminRelevanceService.withdrawConsent(principal.userId, photographerId)

    @DeleteMapping("/photographers/{photographerId}/consent-withdrawal")
    fun clearConsentWithdrawal(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable photographerId: UUID): RelevanceConsentWithdrawalDto =
        adminRelevanceService.clearConsentWithdrawal(principal.userId, photographerId)

    @PostMapping("/alerts/{alertId}/seen")
    fun markSeen(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable alertId: UUID,
    ): RelevanceAlertDto = adminRelevanceService.markSeen(principal.userId, alertId)
}
