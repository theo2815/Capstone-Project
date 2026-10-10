package com.quickpitik.repository

import com.quickpitik.entity.DesktopAuthCode
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.time.OffsetDateTime
import java.util.UUID

@Repository
interface DesktopAuthCodeRepository : JpaRepository<DesktopAuthCode, UUID> {
    fun findByCodeHash(codeHash: String): DesktopAuthCode?

    // Conditional update: two concurrent redeems of one code succeed once. The
    // loser sees 0 rows and is refused by the service.
    @Modifying
    @Query(
        """
        UPDATE DesktopAuthCode c SET c.usedAt = :now
        WHERE c.id = :id AND c.usedAt IS NULL
        """,
    )
    fun markUsed(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int
}
