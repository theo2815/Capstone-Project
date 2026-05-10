package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminFlagDto
import com.quickpitik.dto.admin.FlagActionRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminFlagService
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

@RestController
@RequestMapping("/api/v1/admin/flags")
@PreAuthorize("hasRole('ADMIN')")
class AdminFlagsController(
    private val adminFlagService: AdminFlagService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminFlagDto> =
        adminFlagService.list(status, PaginationParams.of(offset, limit))

    @PostMapping("/{flagId}/resolve")
    fun resolve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @RequestBody(required = false) body: FlagActionRequest?,
    ): AdminFlagDto = adminFlagService.resolve(principal.userId, flagId, body?.resolutionNote)

    @PostMapping("/{flagId}/hide")
    fun hide(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @RequestBody(required = false) body: FlagActionRequest?,
    ): AdminFlagDto = adminFlagService.hide(principal.userId, flagId, body?.resolutionNote)

    @PostMapping("/{flagId}/dismiss")
    fun dismiss(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable flagId: UUID,
        @RequestBody(required = false) body: FlagActionRequest?,
    ): AdminFlagDto = adminFlagService.dismiss(principal.userId, flagId, body?.resolutionNote)
}
