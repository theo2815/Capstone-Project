package com.quickpitik.common

data class PaginatedResponse<T>(
    val items: List<T>,
    val total: Long,
    val offset: Int,
    val limit: Int,
) {
    companion object {
        fun <T> of(items: List<T>, total: Long, params: PaginationParams): PaginatedResponse<T> =
            PaginatedResponse(items = items, total = total, offset = params.offset, limit = params.limit)

        fun <T> empty(params: PaginationParams): PaginatedResponse<T> =
            PaginatedResponse(items = emptyList(), total = 0L, offset = params.offset, limit = params.limit)
    }
}
