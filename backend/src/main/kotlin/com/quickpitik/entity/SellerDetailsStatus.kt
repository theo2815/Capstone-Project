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
        /** The upload gate (AA-I6), its one home: IDENTITY_GATE_ENABLED on = strict, off = legacy (anything but REJECTED). */
        val UPLOAD_GATE_STRICT: Set<SellerDetailsStatus> = setOf(AUTO_APPROVED, APPROVED)
        val UPLOAD_GATE_LEGACY: Set<SellerDetailsStatus> = setOf(MANUAL_REVIEW, AUTO_APPROVED, APPROVED)

        fun uploadGateSet(gateEnabled: Boolean): Set<SellerDetailsStatus> = if (gateEnabled) UPLOAD_GATE_STRICT else UPLOAD_GATE_LEGACY

        /** Parses a `reviewState` value, plus the legacy `"submitted"` (= MANUAL_REVIEW); null when unknown. */
        fun fromWire(value: String): SellerDetailsStatus? =
            if (value == "submitted") MANUAL_REVIEW else entries.firstOrNull { it.reviewState() == value }
    }
}
