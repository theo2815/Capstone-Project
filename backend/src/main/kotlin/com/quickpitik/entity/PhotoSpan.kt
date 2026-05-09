package com.quickpitik.entity

// "default" / "wide" lowercase to match the FE wire shape (`MockPhoto.span`).
// We persist the lowercase form so JPA values map directly without a converter.
enum class PhotoSpan(val wire: String) {
    DEFAULT("default"),
    WIDE("wide"),
    ;

    companion object {
        fun fromWire(value: String): PhotoSpan =
            entries.firstOrNull { it.wire == value } ?: DEFAULT
    }
}
