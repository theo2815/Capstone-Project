package com.quickpitik.service.reference

import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.reference.RaceTypeDto
import com.quickpitik.dto.reference.RaceTypesDto
import com.quickpitik.entity.RaceType
import org.springframework.stereotype.Service

@Service
class RaceTypesService(
    private val races: RaceProperties,
) {
    fun list(): RaceTypesDto = RaceTypesDto(
        selfServeEnabled = races.selfServeEnabled,
        types = RaceType.entries.map { RaceTypeDto(it.name, it.label, it.hint, shared = true) } + OTHER,
    )

    private companion object {
        // "Other" is today's personal flow (event_type NULL); clients send no race type for it.
        val OTHER = RaceTypeDto("OTHER", "Other", "A triathlon, duathlon or any other shoot; your own event", shared = false)
    }
}
