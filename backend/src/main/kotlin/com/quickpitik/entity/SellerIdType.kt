package com.quickpitik.entity

// Government IDs a photographer may give for the seller checks (audit § 6.3: no "other").
// Wire form is lowercase ("drivers_license"); stored uppercase.
enum class SellerIdType {
    PHILSYS,
    DRIVERS_LICENSE,
    PASSPORT,
    UMID,
    PRC,
    POSTAL,
    VOTERS;

    fun toWire(): String = name.lowercase()

    companion object {
        fun fromWire(value: String?): SellerIdType? = entries.firstOrNull { it.toWire() == value }
    }
}
