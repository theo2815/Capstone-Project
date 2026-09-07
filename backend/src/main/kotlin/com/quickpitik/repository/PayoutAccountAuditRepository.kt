package com.quickpitik.repository

import com.quickpitik.entity.PayoutAccountAudit
import org.springframework.data.jpa.repository.JpaRepository
import java.time.OffsetDateTime
import java.util.UUID

interface PayoutAccountAuditRepository : JpaRepository<PayoutAccountAudit, UUID> {
    // Cooldown gate: is there any payout-account change since (now − cooldown)?
    fun existsByUserIdAndCreatedAtAfter(userId: UUID, after: OffsetDateTime): Boolean

    // Latest change — its createdAt + cooldown is the cooldown-until the balance
    // endpoint surfaces to the client.
    fun findFirstByUserIdOrderByCreatedAtDesc(userId: UUID): PayoutAccountAudit?
}
