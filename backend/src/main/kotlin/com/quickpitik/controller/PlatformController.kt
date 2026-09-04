package com.quickpitik.controller

import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.reference.PlatformFeesDto
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/platform")
class PlatformController(
    private val platformProperties: PlatformProperties,
) {
    @GetMapping("/fees")
    fun fees(): PlatformFeesDto = PlatformFeesDto(
        photoPricePhp = platformProperties.photoPricePhp,
        platformCutRate = platformProperties.platformCutRate,
        photographerKeepRate = platformProperties.photographerKeepRate,
        couponMaxPercent = platformProperties.couponMaxPercent,
    )
}
