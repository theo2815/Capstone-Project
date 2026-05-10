package com.quickpitik.entity

// Wire form is lowercase per FE store (`SocialPlatform = "facebook" | ...`).
// JPA persists the enum name; the column has a CHECK constraint that mirrors
// these labels exactly so the DB rejects anything the FE didn't ask for.
enum class SocialPlatform(val wire: String) {
    FACEBOOK("facebook"),
    INSTAGRAM("instagram"),
    TIKTOK("tiktok"),
    X("x"),
    YOUTUBE("youtube"),
    WEBSITE("website");

    companion object {
        fun fromWire(value: String): SocialPlatform? =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
    }
}
