package com.quickpitik.dto.reference

data class RaceTypeDto(val code: String, val label: String, val hint: String, val shared: Boolean)

data class RaceTypesDto(val selfServeEnabled: Boolean, val types: List<RaceTypeDto>)
