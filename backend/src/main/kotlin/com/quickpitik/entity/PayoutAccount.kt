package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One row per payout destination. The single-primary invariant is enforced by
// a partial unique index in V8 (`uq_payout_accounts_primary_per_user`) — JPA
// can't model "exactly one true per group" at the entity level so the DB owns
// it. Service flips primary atomically (clear-then-set in one TX) per Q-E2.
@Entity
@Table(name = "photographer_payout_accounts")
class PayoutAccount(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id", nullable = false, updatable = false)
    val userId: UUID,

    @Enumerated(EnumType.STRING)
    @Column(name = "method", nullable = false, length = 20)
    var method: PayoutMethod,

    @Column(name = "account_number", nullable = false, length = 40)
    var accountNumber: String,

    @Column(name = "account_name", nullable = false, length = 100)
    var accountName: String,

    @Column(name = "qr_s3_key", length = 512)
    var qrS3Key: String? = null,

    @Column(name = "is_primary", nullable = false)
    var isPrimary: Boolean = false,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
)
