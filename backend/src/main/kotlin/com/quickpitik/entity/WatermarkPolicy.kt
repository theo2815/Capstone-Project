package com.quickpitik.entity

// Which marks a photo's preview carries (V46). PLATFORM is the pre-V46
// behaviour and the only policy a PAID event may have; FREE events choose
// OWN (the photographer's logo only) or NONE (plain frame).
enum class WatermarkPolicy(val wire: String) {
    PLATFORM("platform"),
    OWN("own"),
    NONE("none");

    companion object {
        fun fromWire(raw: String?): WatermarkPolicy? =
            entries.firstOrNull { it.wire.equals(raw?.trim(), ignoreCase = true) }
    }
}
