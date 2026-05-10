package com.quickpitik.entity

// Wire form is lowercase per FE store (`PayoutMethod = "gcash" | "maya" | "gotyme"`).
// Kept narrow to the methods the FE actually offers — adding a new method needs
// the DB CHECK constraint, the FE select, and the per-method format validator
// to update together.
enum class PayoutMethod(val wire: String) {
    GCASH("gcash"),
    MAYA("maya"),
    GOTYME("gotyme");

    companion object {
        fun fromWire(value: String): PayoutMethod? =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
    }
}
