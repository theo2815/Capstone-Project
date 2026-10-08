package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalCommand
import com.quickpitik.entity.RemovalCommandState
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import java.time.Instant
import java.util.UUID

interface PhotoRemovalCommandRepository : JpaRepository<PhotoRemovalCommand, UUID> {
    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query(value = """UPDATE photo_removal_commands c SET result = sanitize_photo_removal_result(c.operation, c.result)
        WHERE c.request_id = :requestId AND c.state = 'SUCCEEDED' AND EXISTS
            (SELECT 1 FROM photo_removal_requests r WHERE r.id = c.request_id AND r.status IN ('RESOLVED','CANCELLED'))
        AND c.result IS DISTINCT FROM sanitize_photo_removal_result(c.operation, c.result)""", nativeQuery = true)
    fun sanitizeTerminalResults(requestId: UUID): Int

    @Modifying(flushAutomatically = true)
    @Query(value = """
        INSERT INTO photo_removal_commands
            (command_id, request_id, operation, actor_id_snapshot, input_hash, state, intended_evidence_key, claimed_at)
        VALUES (:commandId, :requestId, :operation, :actorId, :inputHash, 'PENDING', :evidenceKey, :claimedAt)
        ON CONFLICT (command_id) DO NOTHING
    """, nativeQuery = true)
    fun insertClaim(commandId: UUID, requestId: UUID, operation: String, actorId: UUID,
        inputHash: String, evidenceKey: String?, claimedAt: Instant): Int

    fun findByCommandId(commandId: UUID): PhotoRemovalCommand?
    fun findByRequestIdAndStateIn(requestId: UUID, states: Collection<RemovalCommandState>): List<PhotoRemovalCommand>
}
