package com.quickpitik.repository

import com.quickpitik.entity.EmailChangeToken
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.time.OffsetDateTime
import java.util.UUID

@Repository
interface EmailChangeTokenRepository : JpaRepository<EmailChangeToken, UUID> {
    fun findByTokenHash(tokenHash: String): EmailChangeToken?

    // Requesting a new change retires any outstanding ones, so only the newest
    // confirmation link works. Without this, a user who mistypes an address and
    // re-requests leaves a live token pointing at the typo'd inbox.
    @Modifying
    @Query(
        """
        UPDATE EmailChangeToken t SET t.usedAt = :now
        WHERE t.userId = :userId AND t.usedAt IS NULL
        """,
    )
    fun invalidateOutstanding(@Param("userId") userId: UUID, @Param("now") now: OffsetDateTime): Int
}
