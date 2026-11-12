package com.quickpitik.controller

import com.quickpitik.dto.admin.IdentityNoticeResultDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.IdentityNoticeService
import io.swagger.v3.oas.annotations.Operation
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/admin/identity-notice")
@PreAuthorize("hasRole('ADMIN')")
class AdminIdentityNoticeController(
    private val identityNoticeService: IdentityNoticeService,
) {
    @PostMapping
    @Operation(description = "Email photographers who have not passed identity about IDENTITY_GATE_STARTS_ON. dryRun (default true) only counts.")
    fun send(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(defaultValue = "true") dryRun: Boolean,
    ): IdentityNoticeResultDto = identityNoticeService.send(principal.userId, dryRun)
}
