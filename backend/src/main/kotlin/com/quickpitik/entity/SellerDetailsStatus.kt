package com.quickpitik.entity

// Wire form is lowercase; "missing" (no record) exists only on the wire.
enum class SellerDetailsStatus {
    SUBMITTED,
    APPROVED,
    REJECTED;

    fun toWire(): String = name.lowercase()
}
