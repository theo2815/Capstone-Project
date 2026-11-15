package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photographer.FileAppealRequest
import com.quickpitik.dto.photographer.PhotoActionResponse
import com.quickpitik.dto.photographer.PhotoIdsRequest
import com.quickpitik.dto.photographer.PhotographerBlurryPhotoDto
import com.quickpitik.dto.photographer.PhotographerRelevanceCaseDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevancePhotographerHideService
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

// RV-3.2a (A.6): a photographer's own Photo Relevance case, linked from its inbox note, and the appeal of it.
// RV-3.3 (G8): the batch's blurry photos still in the gallery, and the photographer's own hide of them and its undo.
@RestController
@RequestMapping("/api/v1/me/photographer/relevance")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerRelevanceController(
    private val appeals: RelevanceAppealService,
    private val hides: RelevancePhotographerHideService,
) {
    @GetMapping("/cases/{caseId}")
    fun case(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable caseId: UUID): PhotographerRelevanceCaseDto =
        appeals.caseFor(principal.userId, caseId)

    @PostMapping("/cases/{caseId}/appeal")
    fun appeal(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @RequestBody body: FileAppealRequest,
    ): PhotographerRelevanceCaseDto = appeals.file(principal.userId, caseId, body.reason)

    @GetMapping("/cases/{caseId}/blurry")
    fun blurry(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable caseId: UUID,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotographerBlurryPhotoDto> = hides.blurry(principal.userId, caseId, PaginationParams.of(offset, limit))

    // The service checks the 1 to 200 ids (PhotoIdsRequest's own cap is the photo library's 500).
    @PostMapping("/cases/{caseId}/hide")
    fun hide(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable caseId: UUID, @RequestBody body: PhotoIdsRequest): PhotoActionResponse =
        hides.hide(principal.userId, caseId, body.photoIds)

    @PostMapping("/cases/{caseId}/unhide")
    fun unhide(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable caseId: UUID, @RequestBody body: PhotoIdsRequest): PhotoActionResponse =
        hides.unhide(principal.userId, caseId, body.photoIds)
}
