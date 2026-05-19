package com.quickpitik.controller

import com.quickpitik.dto.cart.MergeSavedEventsRequest
import com.quickpitik.dto.cart.RemovedResponse
import com.quickpitik.dto.cart.SaveEventRequest
import com.quickpitik.dto.cart.SavedEventSummaryDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.cart.SavedEventsService
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/saved-events")
class SavedEventsController(
    private val savedEventsService: SavedEventsService,
) {
    @GetMapping
    fun list(@AuthenticationPrincipal principal: AuthPrincipal): List<SavedEventSummaryDto> =
        savedEventsService.list(principal.userId)

    @PostMapping
    fun save(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestBody body: SaveEventRequest,
    ): SavedEventSummaryDto =
        savedEventsService.save(principal.userId, body.eventId)

    @DeleteMapping("/{eventId}")
    fun unsave(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): RemovedResponse =
        RemovedResponse(removed = savedEventsService.unsave(principal.userId, eventId))

    @PostMapping("/merge")
    fun merge(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestBody body: MergeSavedEventsRequest,
    ): List<SavedEventSummaryDto> =
        savedEventsService.merge(principal.userId, body.eventIds)
}
