package com.quickpitik.repository

import com.quickpitik.entity.EmailVerificationToken
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.time.OffsetDateTime
import java.util.UUID

@Repository
interface EmailVerificationTokenRepository : JpaRepository<EmailVerificationToken, UUID> {
    fun findByTokenHash(tokenHash: String): EmailVerificationToken?

    // A resend retires the outstanding link, so only the newest mail works.
    // Without this, every "I didn't get it" click leaves another live 24-hour
    // token behind — mirrors EmailChangeTokenRepository.invalidateOutstanding.
    @Modifying
    @Query(
        """
        UPDATE EmailVerificationToken t SET t.usedAt = :now
        WHERE t.userId = :userId AND t.usedAt IS NULL
        """,
    )
    fun invalidateOutstanding(@Param("userId") userId: UUID, @Param("now") now: OffsetDateTime): Int
}
