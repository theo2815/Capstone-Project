package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// Append-only record of one payout-cycle status transition. Written on EVERY
// transition (request/cancel by photographer, approve/hold/reject/mark-* by
// admin, and later provider webhooks) so the full lifecycle of a payout is
// reconstructable for reconciliation and disputes.
@Entity
@Table(name = "payout_status_history")
class PayoutStatusHistory(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "payout_cycle_id", nullable = false, updatable = false)
    val payoutCycleId: String,

    // Null for the initial transition (cycle creation).
    @Column(name = "from_status")
    val fromStatus: String? = null,

    @Column(name = "to_status", nullable = false)
    val toStatus: String,

    // photographer | admin | system | provider
    @Column(name = "actor_type", nullable = false)
    val actorType: String,

    // The user id behind the transition (photographer or admin); null for
    // system/provider transitions.
    @Column(name = "actor_id")
    val actorId: UUID? = null,

    @Column(name = "reason")
    val reason: String? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
)
