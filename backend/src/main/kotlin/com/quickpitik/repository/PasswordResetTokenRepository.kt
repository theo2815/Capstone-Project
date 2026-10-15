package com.quickpitik.repository

import com.quickpitik.entity.PasswordResetToken
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.time.OffsetDateTime
import java.util.UUID

@Repository
interface PasswordResetTokenRepository : JpaRepository<PasswordResetToken, UUID> {
    fun findByTokenHash(tokenHash: String): PasswordResetToken?

    // Newest-first because legacy pre-V37 rows may leave several outstanding
    // rows per user; only the latest is the live code.
    fun findFirstByUserIdAndUsedAtIsNullOrderByCreatedAtDesc(userId: UUID): PasswordResetToken?

    // Every new request retires the outstanding code, so only the newest mail
    // works. With short numeric codes, N concurrently-live codes would
    // multiply the guess surface — mirrors EmailVerificationTokenRepository.
    @Modifying
    @Query(
        """
        UPDATE PasswordResetToken t SET t.usedAt = :now
        WHERE t.userId = :userId AND t.usedAt IS NULL
        """,
    )
    fun invalidateOutstanding(@Param("userId") userId: UUID, @Param("now") now: OffsetDateTime): Int

    // Wrong guesses this account has used across every reset code created after
    // `since` (security audit 2026-10-14): the per-account budget.
    @Query(
        """
        SELECT COALESCE(SUM(t.attempts), 0) FROM PasswordResetToken t
        WHERE t.userId = :userId AND t.createdAt > :since
        """,
    )
    fun sumAttemptsSince(@Param("userId") userId: UUID, @Param("since") since: OffsetDateTime): Long
}
