package com.quickpitik.controller

import com.quickpitik.dto.photographer.FileAppealRequest
import com.quickpitik.dto.photographer.PhotographerRelevanceCaseDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.relevance.RelevanceAppealService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// RV-3.2a (A.6): a photographer's own Photo Relevance case, linked from its inbox note, and the appeal of it.
@RestController
@RequestMapping("/api/v1/me/photographer/relevance")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerRelevanceController(
    private val appeals: RelevanceAppealService,
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
}
