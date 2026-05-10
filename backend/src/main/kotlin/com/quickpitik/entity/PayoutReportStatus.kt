package com.quickpitik.entity

// Wire form is lowercase per FE PayoutReportStatus. Open is the only state a
// photographer can create directly; admin actions push to acknowledged or
// resolved (Phase G).
enum class PayoutReportStatus(val wire: String) {
    OPEN("open"),
    ACKNOWLEDGED("acknowledged"),
    RESOLVED("resolved");

    companion object {
        fun fromWire(value: String): PayoutReportStatus =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
                ?: throw IllegalArgumentException("Unknown payout report status: $value")
    }
}
