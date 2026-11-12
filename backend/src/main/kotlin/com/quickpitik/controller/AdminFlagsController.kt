package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminFlagDto
import com.quickpitik.dto.admin.FlagActionRequest
import com.quickpitik.dto.admin.HiddenPhotoDto
import com.quickpitik.dto.admin.HideBatchRequest
import com.quickpitik.dto.admin.RelevanceEvidenceDto
import jakarta.validation.Valid
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminFlagService
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

@RestController
@RequestMapping("/api/v1/admin/flags")
@PreAuthorize("hasRole('ADMIN')")
class AdminFlagsController(
    private val adminFlagService: AdminFlagService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) q: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminFlagDto> =
        adminFlagService.list(status, q, PaginationParams.of(offset, limit))

    @PostMapping("/{flagId}/resolve")
    fun resolve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @RequestBody(required = false) body: FlagActionRequest?,
    ): AdminFlagDto = adminFlagService.resolve(principal.userId, flagId, body?.resolutionNote)

    @PostMapping("/{flagId}/hide")
    fun hide(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @RequestBody(required = false) body: FlagActionRequest?,
    ): AdminFlagDto = adminFlagService.hide(principal.userId, flagId, body?.resolutionNote)

    @PostMapping("/{flagId}/dismiss")
    fun dismiss(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @RequestBody(required = false) body: FlagActionRequest?,
    ): AdminFlagDto = adminFlagService.dismiss(principal.userId, flagId, body?.resolutionNote)

    @PostMapping("/{flagId}/escalate")
    fun escalate(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @RequestBody(required = false) body: FlagActionRequest?,
    ): AdminFlagDto = adminFlagService.escalate(principal.userId, flagId, body?.resolutionNote)

    // PE-2.2: confirm a payout hold on a relevance flag. No body.
    @PostMapping("/{flagId}/payout-hold")
    fun confirmPayoutHold(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
    ): AdminFlagDto = adminFlagService.confirmPayoutHold(principal.userId, flagId)

    // PE-2.4: hide every photo of one batch of a relevance flag; the flag stays open.
    @PostMapping("/{flagId}/hide-batch")
    fun hideBatch(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @Valid @RequestBody body: HideBatchRequest,
    ): AdminFlagDto = adminFlagService.hideBatch(principal.userId, flagId, body.eventId!!, body.resolutionNote)

    @GetMapping("/{flagId}/hidden-photos")
    fun hiddenPhotos(
        @PathVariable flagId: UUID,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<HiddenPhotoDto> =
        adminFlagService.listHiddenPhotos(flagId, PaginationParams.of(offset, limit))

    @PostMapping("/{flagId}/hidden-photos/{photoId}/restore")
    fun restoreHiddenPhoto(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @PathVariable photoId: UUID,
    ): AdminFlagDto = adminFlagService.restoreHiddenPhoto(principal.userId, flagId, photoId)

    // PE-2.3: the Low batches behind a relevance flag, with counts and sample photos.
    @GetMapping("/{flagId}/relevance")
    fun relevanceEvidence(@PathVariable flagId: UUID): RelevanceEvidenceDto =
        adminFlagService.relevanceEvidence(flagId)
}
