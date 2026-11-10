package com.quickpitik.controller

import com.quickpitik.dto.report.ReportRequest
import com.quickpitik.dto.report.ReportResultDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.events.RaceReportService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// AA-R9: any signed-in user reports an event or a photographer to the admins.
@RestController
@RequestMapping("/api/v1/me/reports")
@PreAuthorize("isAuthenticated()")
class MeReportsController(
    private val raceReportService: RaceReportService,
    private val rateLimiter: RateLimiter,
) {
    @PostMapping("/events/{eventId}")
    fun reportEvent(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: ReportRequest,
    ): ReportResultDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_REPORT, principal.userId.toString())
        return raceReportService.reportEvent(principal.userId, eventId, body.note)
    }

    @PostMapping("/photographers/{handle}")
    fun reportPhotographer(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable handle: String,
        @Valid @RequestBody body: ReportRequest,
    ): ReportResultDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_REPORT, principal.userId.toString())
        return raceReportService.reportPhotographer(principal.userId, handle, body.note)
    }
}
