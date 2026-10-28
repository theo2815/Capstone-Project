package com.quickpitik.controller

import com.quickpitik.dto.admin.AdminDecisionDto
import com.quickpitik.service.admin.AdminDecisionsService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/admin/decisions")
@PreAuthorize("hasRole('ADMIN')")
class AdminDecisionsController(
    private val adminDecisionsService: AdminDecisionsService,
) {

    @GetMapping
    fun recent(@RequestParam(required = false) limit: Int?): List<AdminDecisionDto> =
        adminDecisionsService.recent(limit)
}
