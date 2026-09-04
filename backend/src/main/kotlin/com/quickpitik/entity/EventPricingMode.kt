package com.quickpitik.entity

// PAID: normal protection + commission. FREE: price 0, no QuickPitik mark,
// originals downloadable by anyone; coupons never apply.
enum class EventPricingMode(val wire: String) {
    PAID("paid"),
    FREE("free");

    companion object {
        fun fromWire(raw: String?): EventPricingMode? =
            entries.firstOrNull { it.wire.equals(raw?.trim(), ignoreCase = true) }
    }
}
