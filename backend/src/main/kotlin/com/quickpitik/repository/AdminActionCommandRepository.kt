package com.quickpitik.repository

import com.quickpitik.entity.AdminActionCommand
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import java.time.Instant
import java.util.UUID

interface AdminActionCommandRepository : JpaRepository<AdminActionCommand, UUID> {
    @Modifying
    @Query(value = """
        INSERT INTO admin_action_commands
            (command_id, actor_id, operation, target_id, input_hash, state, claim_owner, claimed_at, updated_at)
        VALUES (:commandId, :actorId, :operation, :targetId, :inputHash, 'PENDING', :claimOwner, :claimedAt, :claimedAt)
    """, nativeQuery = true)
    fun insertClaim(commandId: UUID, actorId: UUID, operation: String, targetId: UUID?,
        inputHash: String, claimOwner: UUID, claimedAt: Instant): Int
}
