package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// Append-only audit of payout-destination changes. Every create / update /
// set-primary / delete / qr on a photographer_payout_accounts row writes one row
// here, so "who changed where money is sent, and when" is always answerable — and
// the most recent row drives the post-change payout cooldown (no separate column).
// Account numbers are stored MASKED (last 4) — this is a forensic trail, not a
// place to duplicate the sensitive number.
@Entity
@Table(name = "payout_account_audit")
class PayoutAccountAudit(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id", nullable = false, updatable = false)
    val userId: UUID,

    @Column(name = "action", nullable = false, length = 20)
    val action: String,

    @Column(name = "method", length = 20)
    val method: String? = null,

    @Column(name = "account_masked_old", length = 40)
    val accountMaskedOld: String? = null,

    @Column(name = "account_masked_new", length = 40)
    val accountMaskedNew: String? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
)
