package com.quickpitik.common

import com.fasterxml.jackson.annotation.JsonInclude

@JsonInclude(JsonInclude.Include.NON_NULL)
data class ApiError(
    val code: String,
    val message: String,
    val field: String? = null,
)
