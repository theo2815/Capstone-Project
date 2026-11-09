package com.quickpitik.dto.photographer

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.time.LocalDate
import java.util.UUID

// POST /me/photographer/races (AA-R3), bound off the multipart form like CreateMyEventRequest:
// required fields default to "" so a missing part fails @NotBlank on that field.
data class CreateRaceRequest(
    @field:NotBlank
    @field:Size(max = 200)
    val title: String = "",
    @field:NotBlank
    val date: String = "",
    @field:NotBlank
    @field:Size(max = 200)
    val location: String = "",
    @field:NotBlank
    val regionCode: String = "",
    @field:NotBlank
    val provinceCode: String = "",
    // One of the six RaceType codes.
    @field:NotBlank
    val eventType: String = "",
    @field:Size(max = 120)
    val organizerName: String? = null,
    @field:Size(max = 600)
    val description: String? = null,
    // "Mine is different": the candidate the photographer says this race is not.
    val confirmDifferentFrom: UUID? = null,
)

// POST /me/photographer/races/{eventId}/join. typedName = the name the photographer searched for.
data class JoinRaceRequest(
    @field:Size(max = 200)
    val typedName: String? = null,
)

// outcome: created | pending_review | match | candidates | joined (Plan 2 § 3.3).
data class RaceCreateResultDto(
    val outcome: String,
    val event: PhotographerEventDetailDto? = null,
    val candidates: List<RaceCandidateDto> = emptyList(),
    val checkReasons: List<String> = emptyList(),
)

data class RaceCandidateDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val location: String,
    val regionCode: String?,
    val provinceCode: String?,
    val organizerName: String,
    val eventType: String?,
    val bannerUrl: String?,
    val reviewStatus: String,
    // strong | medium
    val strength: String,
)
