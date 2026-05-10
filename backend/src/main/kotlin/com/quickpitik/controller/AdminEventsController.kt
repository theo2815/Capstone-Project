package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminEventDeleteResponseDto
import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminEventService
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

@RestController
@RequestMapping("/api/v1/admin/events")
@PreAuthorize("hasRole('ADMIN')")
class AdminEventsController(
    private val adminEventService: AdminEventService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) state: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
        @RequestParam(required = false) q: String?,
        @RequestParam(required = false) dateFrom: String?,
        @RequestParam(required = false) dateTo: String?,
    ): PaginatedResponse<AdminListEventDto> {
        // q + dateFrom/dateTo are accepted for forwards compatibility with
        // the FE param shape but we filter state in-memory; the heavier
        // server-side filters land when the admin volume justifies it.
        @Suppress("UNUSED_VARIABLE") val _q = q
        @Suppress("UNUSED_VARIABLE") val _dateFrom = dateFrom
        @Suppress("UNUSED_VARIABLE") val _dateTo = dateTo
        return adminEventService.list(state, PaginationParams.of(offset, limit))
    }

    @PostMapping
    fun create(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: CreateAdminEventRequest,
    ): AdminListEventDto = adminEventService.create(principal.userId, body)

    @PatchMapping("/{eventId}")
    fun update(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: UpdateAdminEventRequest,
    ): AdminListEventDto = adminEventService.update(principal.userId, eventId, body)

    @DeleteMapping("/{eventId}")
    fun delete(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): AdminEventDeleteResponseDto = adminEventService.delete(principal.userId, eventId)
}
