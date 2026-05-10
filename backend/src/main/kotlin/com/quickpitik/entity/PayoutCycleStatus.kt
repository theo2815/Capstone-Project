package com.quickpitik.entity

// Wire form is lowercase per FE (`PayoutStatus = "paid" | "pending" | "scheduled"`).
// `held` is admin-only — Phase G uses it to pause a cycle pending an account-info
// fix; the FE renders it under the same chip class as `pending` but the report
// CTA copy changes. Storing as the lowercase wire form keeps DTO mapping a
// no-op pass-through.
enum class PayoutCycleStatus(val wire: String) {
    SCHEDULED("scheduled"),
    PENDING("pending"),
    PAID("paid"),
    HELD("held");

    companion object {
        fun fromWire(value: String): PayoutCycleStatus =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
                ?: throw IllegalArgumentException("Unknown payout cycle status: $value")
    }
}
