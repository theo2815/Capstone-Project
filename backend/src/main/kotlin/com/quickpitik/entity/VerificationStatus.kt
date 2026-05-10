package com.quickpitik.entity

// FE wire shape uses lowercase ("incomplete" | "pending" | "approved"). The
// stored enum name is uppercase per JPA convention; toWire() maps to wire form.
// The store also recognizes REJECTED for admin-side rejection flows (PR 10).
enum class VerificationStatus {
    INCOMPLETE,
    PENDING,
    APPROVED,
    REJECTED;

    fun toWire(): String = when (this) {
        INCOMPLETE -> "incomplete"
        PENDING -> "pending"
        APPROVED -> "approved"
        REJECTED -> "rejected"
    }
}
