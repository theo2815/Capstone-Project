package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminUserDetailDto
import com.quickpitik.dto.admin.AdminUserRowDto
import com.quickpitik.dto.admin.ForceEditUserPatchRequest
import com.quickpitik.dto.admin.RejectUserRequest
import com.quickpitik.dto.admin.SuspendUserRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminUserService
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
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

    @PostMapping("/{userId}/approve")
    fun approve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
    ): AdminUserRowDto = adminUserService.approve(principal.userId, userId)

    @PostMapping("/{userId}/reject")
    fun reject(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: RejectUserRequest,
    ): AdminUserRowDto = adminUserService.reject(principal.userId, userId, body.reason)

    @PostMapping("/{userId}/reset-verification")
    fun resetVerification(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
    ): AdminUserRowDto = adminUserService.resetVerification(principal.userId, userId)

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

    @PatchMapping("/{userId}")
    fun forceEdit(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: ForceEditUserPatchRequest,
    ): AdminUserRowDto = adminUserService.forceEdit(principal.userId, userId, body)
}
