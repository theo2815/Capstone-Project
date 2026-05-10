package com.quickpitik.entity

// Wire form is lowercase snake_case to match
// website/src/lib/admin-payout-reports.ts PayoutReportReason. Service maps the
// FE-submitted reason via fromWire and returns INVALID_REASON when nothing
// matches — the V9 migration's CHECK constraint is the second line of defence.
enum class PayoutReportReason(val wire: String) {
    NOT_RECEIVED("not_received"),
    WRONG_AMOUNT("wrong_amount"),
    ACCOUNT_INFO("account_info"),
    PROCESSING_DELAY("processing_delay"),
    OTHER("other");

    companion object {
        fun fromWire(value: String): PayoutReportReason? =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
    }
}
