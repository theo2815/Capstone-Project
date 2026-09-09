package com.quickpitik.entity

// Wire form is lowercase. This enum MUST stay a superset of the DB status CHECK
// (payout_cycles_status_check) so fromWire never throws on a valid row.
//
// Lifecycle (auditable payout state machine):
//   pending(_review) → approved → processing → paid           (happy path)
//                    ↘ held ⇄ approved                        (admin pause)
//                    ↘ rejected                               (admin denies)
//   photographer may cancel a held request → cancelled
//   provider outcomes after processing → failed | returned    (reservation released)
// PENDING is the request-based "pending review"; scheduled is the legacy weekly
// model (kept only so pre-existing rows still parse). APPROVED/PROCESSING/FAILED/
// RETURNED/REJECTED are wired by the admin lifecycle phase; CANCELLED is live now
// (soft-cancel of a withdrawn request). "Open" (reserves funds) = scheduled,
// pending, approved, processing, held; the rest are terminal.
enum class PayoutCycleStatus(val wire: String) {
    SCHEDULED("scheduled"),
    PENDING("pending"),
    APPROVED("approved"),
    PROCESSING("processing"),
    PAID("paid"),
    FAILED("failed"),
    RETURNED("returned"),
    HELD("held"),
    REJECTED("rejected"),
    CANCELLED("cancelled");

    companion object {
        fun fromWire(value: String): PayoutCycleStatus =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
                ?: throw IllegalArgumentException("Unknown payout cycle status: $value")
    }
}
