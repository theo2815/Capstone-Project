package com.quickpitik.entity

// Wire form is lowercase, matching the other status enums (PayoutCycleStatus,
// PayoutMethod, ...) so DTO mapping stays a pass-through.
//
// Two states by construction. There is no `rejected`: SelfieService.qualityGate
// throws before the row is written, so a rejected selfie never reaches the
// database — the runner gets a SELFIE_REJECTED envelope instead. `untested` is
// what AI_API_ENABLED=false produces; the selfie is stored and still usable for
// face search, it just never went through the ai-api quality gate.
enum class SelfieQualityTestStatus(val wire: String) {
    UNTESTED("untested"),
    PASSED("passed");

    companion object {
        fun fromWire(value: String): SelfieQualityTestStatus =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
                ?: throw IllegalArgumentException("Unknown selfie quality test status: $value")
    }
}
