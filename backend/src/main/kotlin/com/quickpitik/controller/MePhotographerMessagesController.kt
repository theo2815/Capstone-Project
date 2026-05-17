package com.quickpitik.controller

import com.quickpitik.dto.photographer.MarkAllReadResponse
import com.quickpitik.dto.photographer.MessageRemovedResponse
import com.quickpitik.dto.photographer.PhotographerMessageDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.photographer.PhotographerMessageService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// Photographer-facing inbox. V10 created the table + writers; V15 added
// the columns the FE needs (title + removed_at) and the admin_message
// kind. This controller closes the loop so the photographer's bell +
// inbox modal can read what every admin action has been pushing
// server-side since PR 10.
@RestController
@RequestMapping("/api/v1/me/photographer/messages")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerMessagesController(
    private val photographerMessageService: PhotographerMessageService,
) {
    @GetMapping
    fun list(
        @AuthenticationPrincipal principal: AuthPrincipal,
    ): List<PhotographerMessageDto> =
        photographerMessageService.list(principal.userId)

    @PatchMapping("/{id}/read")
    fun markRead(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): PhotographerMessageDto =
        photographerMessageService.markRead(principal.userId, id)

    @PatchMapping("/read-all")
    fun markAllRead(
        @AuthenticationPrincipal principal: AuthPrincipal,
    ): MarkAllReadResponse =
        photographerMessageService.markAllRead(principal.userId)

    @DeleteMapping("/{id}")
    fun remove(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): MessageRemovedResponse =
        photographerMessageService.remove(principal.userId, id)
}
