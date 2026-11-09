package com.quickpitik.controller

import com.quickpitik.dto.reference.RaceTypesDto
import com.quickpitik.service.reference.RaceTypesService
import org.springframework.http.CacheControl
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.time.Duration

@RestController
@RequestMapping("/api/v1/race-types")
class RaceTypesController(
    private val raceTypesService: RaceTypesService,
) {
    // Short cache: selfServeEnabled follows the rollout switch.
    @GetMapping
    fun all(): ResponseEntity<RaceTypesDto> =
        ResponseEntity.ok()
            .cacheControl(CacheControl.maxAge(Duration.ofMinutes(5)).cachePublic())
            .body(raceTypesService.list())
}
