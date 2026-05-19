package com.quickpitik.controller

import com.quickpitik.dto.photographer.MarkAllReadResponse
import com.quickpitik.dto.photographer.MessageRemovedResponse
import com.quickpitik.dto.runner.RunnerMessageDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.runner.RunnerMessageReaderService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// Runner-facing inbox. Mirrors MePhotographerMessagesController exactly
// except for the role gate + DTO shape (RunnerMessageDto carries orderId
// for the dispute-outcome deep-link to /orders).
@RestController
@RequestMapping("/api/v1/me/runner/messages")
@PreAuthorize("hasRole('RUNNER')")
class MeRunnerMessagesController(
    private val runnerMessageReaderService: RunnerMessageReaderService,
) {
    @GetMapping
    fun list(
        @AuthenticationPrincipal principal: AuthPrincipal,
    ): List<RunnerMessageDto> =
        runnerMessageReaderService.list(principal.userId)

    @PatchMapping("/{id}/read")
    fun markRead(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): RunnerMessageDto =
        runnerMessageReaderService.markRead(principal.userId, id)

    @PatchMapping("/read-all")
    fun markAllRead(
        @AuthenticationPrincipal principal: AuthPrincipal,
    ): MarkAllReadResponse =
        runnerMessageReaderService.markAllRead(principal.userId)

    @DeleteMapping("/{id}")
    fun remove(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): MessageRemovedResponse =
        runnerMessageReaderService.remove(principal.userId, id)
}
