package com.quickpitik.common

import org.springframework.data.domain.Pageable
import org.springframework.data.domain.Sort

// Page-based PageRequest only honors offsets that are multiples of pageSize, but
// the FE contract (Q-017) speaks in arbitrary offset/limit pairs. This adapter
// exposes the absolute offset so Hibernate uses setFirstResult() literally.
class OffsetLimitPageable(
    private val offsetValue: Long,
    private val limitValue: Int,
    private val sortValue: Sort = Sort.unsorted(),
) : Pageable {

    constructor(params: PaginationParams, sort: Sort = Sort.unsorted()) :
        this(offsetValue = params.offset.toLong(), limitValue = params.limit, sortValue = sort)

    override fun getPageNumber(): Int = (offsetValue / limitValue).toInt()

    override fun getPageSize(): Int = limitValue

    override fun getOffset(): Long = offsetValue

    override fun getSort(): Sort = sortValue

    override fun next(): Pageable =
        OffsetLimitPageable(offsetValue + limitValue, limitValue, sortValue)

    override fun previousOrFirst(): Pageable =
        if (offsetValue <= limitValue) first() else OffsetLimitPageable(offsetValue - limitValue, limitValue, sortValue)

    override fun first(): Pageable = OffsetLimitPageable(0, limitValue, sortValue)

    override fun withPage(pageNumber: Int): Pageable =
        OffsetLimitPageable(pageNumber.toLong() * limitValue, limitValue, sortValue)

    override fun hasPrevious(): Boolean = offsetValue > 0

    override fun isPaged(): Boolean = true
}
