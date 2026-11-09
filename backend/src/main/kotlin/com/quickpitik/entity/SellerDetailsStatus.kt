package com.quickpitik.entity

// "missing" (no record) exists only on the wire.
enum class SellerDetailsStatus {
    MANUAL_REVIEW,
    AUTO_APPROVED,
    APPROVED,
    REJECTED;

    /**
     * The `status` field keeps the values clients knew before V94 (submitted / approved / rejected), so
     * today's website slab and admin queue, an older Vercel build and old Android builds keep working.
     */
    fun toWire(): String = when (this) {
        MANUAL_REVIEW -> "submitted"
        AUTO_APPROVED -> "approved"
        else -> name.lowercase()
    }

    /** The detailed state, sent as `reviewState`; new clients read only this. */
    fun reviewState(): String = name.lowercase()

    /** Identity counts as checked: passed automatically or approved by an admin. */
    fun passesIdentity(): Boolean = this == AUTO_APPROVED || this == APPROVED

    companion object {
        /** Parses a `reviewState` value, plus the legacy `"submitted"` (= MANUAL_REVIEW); null when unknown. */
        fun fromWire(value: String): SellerDetailsStatus? =
            if (value == "submitted") MANUAL_REVIEW else entries.firstOrNull { it.reviewState() == value }
    }
}
