package com.quickpitik.repository

import com.quickpitik.entity.RefreshToken
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.time.OffsetDateTime
import java.util.UUID

@Repository
interface RefreshTokenRepository : JpaRepository<RefreshToken, UUID> {
    fun findByTokenHash(tokenHash: String): RefreshToken?

    @Modifying
    @Query(
        """
        UPDATE RefreshToken rt
        SET rt.revokedAt = :now
        WHERE rt.userId = :userId AND rt.revokedAt IS NULL
        """,
    )
    fun revokeAllForUser(@Param("userId") userId: UUID, @Param("now") now: OffsetDateTime): Int
}
