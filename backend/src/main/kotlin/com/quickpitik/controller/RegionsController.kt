package com.quickpitik.controller

import com.quickpitik.dto.reference.RegionDto
import com.quickpitik.service.reference.RegionsService
import org.springframework.http.CacheControl
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.time.Duration

@RestController
@RequestMapping("/api/v1/regions")
class RegionsController(
    private val regionsService: RegionsService,
) {
    @GetMapping
    fun all(): ResponseEntity<List<RegionDto>> =
        ResponseEntity.ok()
            .cacheControl(CacheControl.maxAge(Duration.ofDays(1)).cachePublic())
            .body(regionsService.all())
}
