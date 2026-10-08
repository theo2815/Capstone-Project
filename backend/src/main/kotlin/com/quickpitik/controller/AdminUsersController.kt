package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminPhotographerSettingsDto
import com.quickpitik.dto.admin.AdminUserDetailDto
import com.quickpitik.dto.admin.AdminUserRowDto
import com.quickpitik.dto.admin.ApproveUserRequest
import com.quickpitik.dto.admin.RejectUserRequest
import com.quickpitik.dto.admin.ResetVerificationRequest
import com.quickpitik.dto.admin.SuspendUserRequest
import com.quickpitik.dto.photographer.AdminUserMessageRequest
import com.quickpitik.dto.photographer.PhotographerMessageDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminUserService
import jakarta.validation.Valid
import io.swagger.v3.oas.annotations.Operation
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
@RequestMapping("/api/v1/admin/users")
@PreAuthorize("hasRole('ADMIN')")
class AdminUsersController(
    private val adminUserService: AdminUserService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) role: String?,
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) q: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminUserRowDto> =
        adminUserService.list(role, status, q, PaginationParams.of(offset, limit))

    @GetMapping("/{userId}")
    fun detail(@PathVariable userId: UUID): AdminUserDetailDto =
        adminUserService.detail(userId)

    @GetMapping("/{userId}/settings")
    fun photographerSettings(@PathVariable userId: UUID): AdminPhotographerSettingsDto =
        adminUserService.photographerSettings(userId)

    @PostMapping("/{userId}/approve")
    @Operation(description = "Approve a complete PENDING photographer. Optional expectedRevision rejects stale review generations with 409; omitting the body remains supported for v1 callers.")
    fun approve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody(required = false) body: ApproveUserRequest? = null,
    ): AdminUserRowDto = adminUserService.approve(principal.userId, userId, body?.expectedRevision)

    @PostMapping("/{userId}/reject")
    @Operation(description = "Reject a PENDING photographer. Optional expectedRevision rejects stale review generations with 409.")
    fun reject(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: RejectUserRequest,
    ): AdminUserRowDto = adminUserService.reject(principal.userId, userId, body.reason, body.expectedRevision)

    @PostMapping("/{userId}/reset-verification")
    @Operation(description = "Reset a PENDING, APPROVED or legacy REJECTED photographer. Optional expectedRevision rejects stale review generations with 409.")
    fun resetVerification(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: ResetVerificationRequest,
    ): AdminUserRowDto = adminUserService.resetVerification(principal.userId, userId, body.reason, body.expectedRevision)

    @PostMapping("/{userId}/suspend")
    fun suspend(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: SuspendUserRequest,
    ): AdminUserRowDto = adminUserService.suspend(principal.userId, userId, body.reason)

    @PostMapping("/{userId}/unsuspend")
    fun unsuspend(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
    ): AdminUserRowDto = adminUserService.unsuspend(principal.userId, userId)

    @PostMapping("/{userId}/message")
    fun sendMessage(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: AdminUserMessageRequest,
    ): PhotographerMessageDto =
        adminUserService.sendMessage(principal.userId, userId, body.subject, body.body)
}
