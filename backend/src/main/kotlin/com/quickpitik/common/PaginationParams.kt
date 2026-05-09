package com.quickpitik.common

data class PaginationParams(
    val offset: Int,
    val limit: Int,
) {
    companion object {
        const val DEFAULT_LIMIT = 50
        const val MAX_LIMIT = 200

        fun of(offset: Int?, limit: Int?): PaginationParams {
            val o = (offset ?: 0).coerceAtLeast(0)
            val l = (limit ?: DEFAULT_LIMIT).coerceIn(1, MAX_LIMIT)
            return PaginationParams(o, l)
        }
    }
}
