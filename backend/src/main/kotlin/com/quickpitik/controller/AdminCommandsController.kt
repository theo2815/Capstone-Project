package com.quickpitik.controller

import com.quickpitik.common.IdempotencyKey
import com.quickpitik.dto.admin.AdminActionCommandDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminActionCommandService
import io.swagger.v3.oas.annotations.Operation
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/admin/commands")
@PreAuthorize("hasRole('ADMIN')")
class AdminCommandsController(
    private val commands: AdminActionCommandService,
) {

    @GetMapping("/{commandId}")
    @Operation(description = "Look up what happened to one of the caller's own keyed admin requests (the UUID sent as Idempotency-Key). Returns the receipt state (PENDING, UNKNOWN, SUCCEEDED or FAILED) and result IDs only. Another admin's key and an unknown key both return 404; a malformed key returns 400.")
    fun find(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable commandId: String,
    ): AdminActionCommandDto = commands.find(principal.userId, IdempotencyKey.parse(commandId))
}
