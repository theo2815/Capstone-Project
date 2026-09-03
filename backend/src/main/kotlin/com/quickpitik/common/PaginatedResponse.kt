package com.quickpitik.common

import com.fasterxml.jackson.annotation.JsonInclude
import java.time.OffsetDateTime

@JsonInclude(JsonInclude.Include.NON_NULL)
data class PaginatedResponse<T>(
    val items: List<T>,
    val total: Long,
    val offset: Int,
    val limit: Int,
    val snapshotAt: OffsetDateTime? = null,
) {
    companion object {
        fun <T> of(items: List<T>, total: Long, params: PaginationParams): PaginatedResponse<T> =
            PaginatedResponse(items = items, total = total, offset = params.offset, limit = params.limit)

        fun <T> empty(params: PaginationParams): PaginatedResponse<T> =
            PaginatedResponse(items = emptyList(), total = 0L, offset = params.offset, limit = params.limit)
    }
}
