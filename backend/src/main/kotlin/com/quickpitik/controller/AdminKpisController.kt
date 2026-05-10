package com.quickpitik.controller

import com.quickpitik.dto.admin.AdminKpisDto
import com.quickpitik.dto.admin.AdminTrendPointDto
import com.quickpitik.service.admin.AdminKpiService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/admin")
@PreAuthorize("hasRole('ADMIN')")
class AdminKpisController(
    private val adminKpiService: AdminKpiService,
) {

    @GetMapping("/kpis")
    fun kpis(): AdminKpisDto = adminKpiService.kpis()

    @GetMapping("/kpis/trend")
    fun trend(@RequestParam(required = false) days: Int?): List<AdminTrendPointDto> =
        adminKpiService.trend(days)
}
