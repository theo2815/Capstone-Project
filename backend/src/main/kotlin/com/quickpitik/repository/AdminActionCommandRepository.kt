package com.quickpitik.repository

import org.springframework.data.jpa.repository.Lock
import jakarta.persistence.LockModeType
import com.quickpitik.entity.AdminActionCommand
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import java.time.Instant
import java.util.UUID

interface AdminActionCommandRepository : JpaRepository<AdminActionCommand, UUID> {
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select c from AdminActionCommand c where c.commandId = :commandId")
    fun findForUpdate(commandId: UUID): AdminActionCommand?

    @Modifying
    @Query(value = """
        INSERT INTO admin_action_commands
            (command_id, actor_id, operation, target_id, input_hash, state, claim_owner, claimed_at, updated_at,
             lease_expires_at, intended_event_id, intended_storage_key, processed_cover_digest)
        VALUES (:commandId, :actorId, :operation, :targetId, :inputHash, 'PENDING', :owner, :now, :now,
                :expires, :eventId, :storageKey, :digest)
        ON CONFLICT (command_id) DO NOTHING
    """, nativeQuery = true)
    fun insertEventClaim(commandId: UUID, actorId: UUID, operation: String, targetId: UUID?, inputHash: String,
        owner: UUID, now: Instant, expires: Instant, eventId: UUID, storageKey: String?, digest: String?): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(value = """
        UPDATE admin_action_commands SET claim_owner = :newOwner, lease_expires_at = :expires,
            state = 'PENDING', updated_at = :now
        WHERE command_id = :commandId AND claim_owner = :oldOwner AND lease_expires_at <= :now
            AND state IN ('PENDING', 'UNKNOWN')
    """, nativeQuery = true)
    fun takeOver(commandId: UUID, oldOwner: UUID, newOwner: UUID, now: Instant, expires: Instant): Int

    @Modifying
    @Query(value = """
        INSERT INTO admin_action_commands
            (command_id, actor_id, operation, target_id, input_hash, state, claim_owner, claimed_at, updated_at)
        VALUES (:commandId, :actorId, :operation, :targetId, :inputHash, 'PENDING', :claimOwner, :claimedAt, :claimedAt)
    """, nativeQuery = true)
    fun insertClaim(commandId: UUID, actorId: UUID, operation: String, targetId: UUID?,
        inputHash: String, claimOwner: UUID, claimedAt: Instant): Int
}
