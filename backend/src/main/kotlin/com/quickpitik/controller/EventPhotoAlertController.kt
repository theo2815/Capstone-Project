package com.quickpitik.controller

import com.quickpitik.dto.cart.RemovedResponse
import com.quickpitik.dto.events.PhotoAlertRequest
import com.quickpitik.dto.events.PhotoAlertStatusDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.events.EventPhotoAlertService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

// Runner-only opt-in for the "your photos are ready" email. Cheap DB upsert —
// no rate limiter; the ai-api match happens later in EventPhotosReadySweep.
// Shares the /api/v1/events base path with EventPhotoController; the
// /{slug}/photo-alert sub-paths don't collide with /{slug}/photos.
@RestController
@RequestMapping("/api/v1/events")
@PreAuthorize("hasRole('RUNNER')")
class EventPhotoAlertController(
    private val service: EventPhotoAlertService,
) {
    @PostMapping("/{slug}/photo-alert")
    fun register(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable slug: String,
        @RequestBody body: PhotoAlertRequest,
    ): PhotoAlertStatusDto =
        service.register(principal.userId, slug, body.selfieId)

    @DeleteMapping("/{slug}/photo-alert")
    fun optOut(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable slug: String,
    ): RemovedResponse =
        RemovedResponse(removed = service.optOut(principal.userId, slug))

    @GetMapping("/{slug}/photo-alert")
    fun status(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable slug: String,
    ): PhotoAlertStatusDto =
        service.status(principal.userId, slug)
}
