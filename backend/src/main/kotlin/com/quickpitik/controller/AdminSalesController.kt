package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminSalesEventRowDto
import com.quickpitik.dto.admin.AdminSalesKpisDto
import com.quickpitik.dto.admin.AdminSalesTrendDto
import com.quickpitik.dto.admin.AdminTopPhotographersDto
import com.quickpitik.service.admin.AdminSalesService
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/admin/sales")
@PreAuthorize("hasRole('ADMIN')")
class AdminSalesController(
    private val adminSalesService: AdminSalesService,
) {

    @GetMapping("/kpis")
    fun kpis(@RequestParam(required = false) range: String?): AdminSalesKpisDto =
        adminSalesService.kpis(range)

    @GetMapping("/trend")
    fun trend(@RequestParam(required = false) weeks: Int?): AdminSalesTrendDto =
        adminSalesService.trend(weeks)

    @GetMapping("/top-photographers")
    fun topPhotographers(
        @RequestParam(required = false) range: String?,
        @RequestParam(required = false) limit: Int?,
    ): AdminTopPhotographersDto =
        adminSalesService.topPhotographers(range, limit)

    @GetMapping("/by-event")
    fun byEvent(
        @RequestParam(required = false) order: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminSalesEventRowDto> =
        adminSalesService.byEvent(order, PaginationParams.of(offset, limit))
}
