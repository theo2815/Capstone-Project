package com.quickpitik.repository

import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.LivenessSession
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

interface LivenessSessionRepository : JpaRepository<LivenessSession, String> {
    /** Every purpose counts toward one per-user daily cap. */
    fun countByUserIdAndCreatedAtGreaterThanEqual(userId: UUID, since: OffsetDateTime): Long

    fun findBySessionIdAndUserIdAndPurpose(sessionId: String, userId: UUID, purpose: LivenessPurpose): LivenessSession?

    /** Single use: 1 when this call consumed the session, 0 when it is unknown, someone else's, for another purpose or event, or used. */
    @Modifying
    @Query(
        value = """UPDATE liveness_sessions SET consumed_at = :now
            WHERE session_id = :sessionId AND user_id = :userId AND purpose = :purpose
              AND event_id IS NOT DISTINCT FROM CAST(:eventId AS uuid) AND consumed_at IS NULL""",
        nativeQuery = true,
    )
    fun claim(
        @Param("sessionId") sessionId: String,
        @Param("userId") userId: UUID,
        @Param("purpose") purpose: String,
        @Param("eventId") eventId: UUID?,
        @Param("now") now: OffsetDateTime,
    ): Int
}
