package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminPayoutCycleDto
import com.quickpitik.dto.admin.BulkPayoutResultDto
import com.quickpitik.dto.admin.BulkPayoutsRequest
import com.quickpitik.dto.admin.GenerateCyclesResultDto
import com.quickpitik.dto.admin.HoldPayoutRequest
import com.quickpitik.dto.admin.MarkPayoutPaidRequest
import com.quickpitik.exception.ValidationException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminPayoutService
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestHeader
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeParseException

@RestController
@RequestMapping("/api/v1/admin/payouts")
@PreAuthorize("hasRole('ADMIN')")
class AdminPayoutsController(
    private val adminPayoutService: AdminPayoutService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) q: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminPayoutCycleDto> =
        adminPayoutService.list(status, q, PaginationParams.of(offset, limit))

    @PostMapping("/{payoutId}/approve")
    fun approve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable payoutId: String,
    ): AdminPayoutCycleDto = adminPayoutService.approve(principal.userId, payoutId)

    @PostMapping("/{payoutId}/hold")
    fun hold(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable payoutId: String,
        @Valid @RequestBody body: HoldPayoutRequest,
    ): AdminPayoutCycleDto = adminPayoutService.hold(principal.userId, payoutId, body.reason)

    @PostMapping("/{payoutId}/mark-paid")
    fun markPaid(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable payoutId: String,
        @Valid @RequestBody body: MarkPayoutPaidRequest,
    ): AdminPayoutCycleDto = adminPayoutService.markPaid(principal.userId, payoutId, body)

    // Idempotent on the inbound Idempotency-Key header (RFC 9110 §9.2.2,
    // matches POST /api/v1/orders). `required = false` so our own
    // IdempotencyKey.parse can produce the canonical 400
    // INVALID_IDEMPOTENCY_KEY envelope on missing / empty / bad-UUID input;
    // `required = true` would surface as a generic 500 via the catch-all
    // advice. C-3.
    @PostMapping("/bulk")
    fun bulk(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestHeader(name = IdempotencyKey.HEADER, required = false) idempotencyHeader: String?,
        @Valid @RequestBody body: BulkPayoutsRequest,
    ): BulkPayoutResultDto {
        val key = IdempotencyKey.parse(idempotencyHeader)?.value
            ?: throw ValidationException(
                code = ErrorCodes.INVALID_IDEMPOTENCY_KEY,
                message = "Idempotency-Key header is required and must be a UUID",
                field = IdempotencyKey.HEADER,
            )
        return adminPayoutService.bulk(principal.userId, key, body)
    }

    // Admin-triggered cycle generator. Accepts `weekOf` (Monday YYYY-MM-DD);
    // when omitted, defaults to the previous ISO Monday in Asia/Manila so the
    // common case ("generate last week") is a one-click button. Idempotent —
    // see AdminPayoutService.generateForWeek.
    @PostMapping("/generate")
    fun generate(
        @RequestParam(required = false) weekOf: String?,
    ): GenerateCyclesResultDto {
        val resolvedWeek = weekOf
            ?.takeIf { it.isNotBlank() }
            ?.let {
                try {
                    LocalDate.parse(it.trim())
                } catch (_: DateTimeParseException) {
                    throw ValidationException(
                        code = ErrorCodes.VALIDATION_ERROR,
                        message = "weekOf must be ISO date (YYYY-MM-DD)",
                        field = "weekOf",
                    )
                }
            }
            ?: previousIsoMonday()
        return adminPayoutService.generateForWeek(resolvedWeek)
    }

    private fun previousIsoMonday(): LocalDate {
        val today = LocalDate.now(ZoneId.of("Asia/Manila"))
        val thisMonday = today.minusDays((today.dayOfWeek.value - DayOfWeek.MONDAY.value).toLong())
        return thisMonday.minusWeeks(1)
    }
}
