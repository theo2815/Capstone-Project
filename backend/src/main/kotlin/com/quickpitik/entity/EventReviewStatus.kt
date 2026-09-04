package com.quickpitik.entity

// Admin review state of a photographer-owned event (V46). Admin-created
// events are APPROVED from birth. PENDING/REJECTED events sit at
// EventStatus.DRAFT (invisible, no uploads); CHANGE_PENDING is a LIVE event
// whose owner has requested a pricing change that only an admin can apply.
enum class EventReviewStatus(val wire: String) {
    PENDING("pending"),
    APPROVED("approved"),
    REJECTED("rejected"),
    CHANGE_PENDING("change_pending");

    val inQueue: Boolean get() = this == PENDING || this == CHANGE_PENDING
}
