package com.quickpitik.controller

import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photographer.MarkAllReadResponse
import com.quickpitik.dto.photographer.MessageRemovedResponse
import com.quickpitik.dto.photographer.PhotographerMessageDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.photographer.PhotographerMessageService
import jakarta.servlet.http.HttpServletResponse
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
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
    // Paged since 2026-08-14 — this used to return every message the
    // photographer had ever received. Response stays a bare JSON array (no
    // envelope change), so existing web + mobile callers that send no params
    // keep working; they just cap at MESSAGES_DEFAULT_LIMIT. The true un-removed
    // total rides the X-Total-Count header (CORS-exposed) for the web inbox.
    @GetMapping
    fun list(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
        response: HttpServletResponse,
    ): List<PhotographerMessageDto> {
        response.setHeader(
            "X-Total-Count",
            photographerMessageService.count(principal.userId).toString(),
        )
        return photographerMessageService.list(
            photographerId = principal.userId,
            params = PaginationParams.of(offset, limit ?: MESSAGES_DEFAULT_LIMIT),
        )
    }

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

    private companion object {
        const val MESSAGES_DEFAULT_LIMIT = 100
    }
}
