package com.quickpitik.dto.reference

data class RegionDto(
    val code: String,
    val name: String,
    val shortName: String,
    val group: String,
    val provinces: List<ProvinceDto>,
)

data class ProvinceDto(
    val code: String,
    val name: String,
)
