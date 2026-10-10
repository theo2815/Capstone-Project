package com.quickpitik.support

import com.quickpitik.entity.DesktopAuthCode
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.DesktopAuthCodeRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// Pins the one JPQL update the desktop connect flow relies on: a code is marked
// used exactly once, so two concurrent redeems cannot both succeed (backend
// AGENTS.md rule 13: every new @Query gets a Postgres test).
class DesktopAuthCodeRepositoryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var codes: DesktopAuthCodeRepository

    @Autowired
    private lateinit var users: UserRepository

    @Autowired
    private lateinit var transactionTemplate: TransactionTemplate

    @Test
    fun `markUsed updates a live code once and never twice`() {
        val user = users.save(
            User(
                email = "desktop-${UUID.randomUUID()}@test.local",
                passwordHash = "unused",
                name = "Desktop Repo Test",
                role = Role.PHOTOGRAPHER,
            ),
        )
        val now = OffsetDateTime.now()
        val code = codes.save(
            DesktopAuthCode(
                userId = user.id,
                codeHash = "hash-${UUID.randomUUID()}",
                codeChallenge = "c".repeat(43),
                expiresAt = now.plusMinutes(2),
                createdAt = now,
            ),
        )
        assertNull(codes.findById(code.id).get().usedAt)

        val first = transactionTemplate.execute { codes.markUsed(code.id, now) }
        val second = transactionTemplate.execute { codes.markUsed(code.id, now) }

        assertEquals(1, first)
        assertEquals(0, second)
        assertNotNull(codes.findById(code.id).get().usedAt)
        assertEquals(code.id, codes.findByCodeHash(code.codeHash)?.id)
    }
}
