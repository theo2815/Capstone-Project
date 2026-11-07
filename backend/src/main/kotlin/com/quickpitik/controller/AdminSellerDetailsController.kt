package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminSellerDetailsDto
import com.quickpitik.dto.admin.AdminSellerDetailsRowDto
import com.quickpitik.dto.admin.SellerDetailsApproveRequest
import com.quickpitik.dto.admin.SellerDetailsRejectRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.photographer.SellerDetailsService
import jakarta.servlet.http.HttpServletResponse
import jakarta.validation.Valid
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

// SC-1: admins review photographers' seller details (legal name, ID photos, selfie, address, phone).
@RestController
@RequestMapping("/api/v1/admin/seller-details")
@PreAuthorize("hasRole('ADMIN')")
class AdminSellerDetailsController(
    private val sellerDetailsService: SellerDetailsService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminSellerDetailsRowDto> =
        sellerDetailsService.adminList(status, PaginationParams.of(offset, limit))

    @GetMapping("/{userId}")
    fun view(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        response: HttpServletResponse,
    ): AdminSellerDetailsDto {
        response.setHeader("Cache-Control", "private, no-store")
        return sellerDetailsService.adminView(principal.userId, userId)
    }

    @PostMapping("/{userId}/approve")
    fun approve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: SellerDetailsApproveRequest,
    ): AdminSellerDetailsDto = sellerDetailsService.approve(principal.userId, userId, body.revision!!)

    @PostMapping("/{userId}/reject")
    fun reject(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable userId: UUID,
        @Valid @RequestBody body: SellerDetailsRejectRequest,
    ): AdminSellerDetailsDto = sellerDetailsService.reject(principal.userId, userId, body.revision!!, body.reason)
}
