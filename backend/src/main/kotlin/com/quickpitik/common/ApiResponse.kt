package com.quickpitik.common

import com.fasterxml.jackson.annotation.JsonInclude

@JsonInclude(JsonInclude.Include.NON_NULL)
data class ApiResponse<T>(
    val success: Boolean,
    val data: T? = null,
    val errors: List<ApiError>? = null,
) {
    companion object {
        fun <T> success(data: T?): ApiResponse<T> =
            ApiResponse(success = true, data = data)

        fun failure(error: ApiError): ApiResponse<Nothing> =
            ApiResponse(success = false, errors = listOf(error))

        fun failure(errors: List<ApiError>): ApiResponse<Nothing> =
            ApiResponse(success = false, errors = errors)
    }
}
