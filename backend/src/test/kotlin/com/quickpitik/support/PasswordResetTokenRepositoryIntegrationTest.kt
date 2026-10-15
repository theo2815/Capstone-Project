package com.quickpitik.support

import com.quickpitik.entity.PasswordResetToken
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.PasswordResetTokenRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

// Pins sumAttemptsSince, the per-account reset-code budget (security audit
// 2026-10-14; backend AGENTS.md rule 13: every new @Query gets a Postgres test).
class PasswordResetTokenRepositoryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var tokens: PasswordResetTokenRepository

    @Autowired
    private lateinit var users: UserRepository

    private fun user() = users.save(
        User(email = "reset-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Reset Repo Test", role = Role.RUNNER),
    )

    private fun row(userId: UUID, attempts: Int, createdAt: OffsetDateTime) = tokens.save(
        PasswordResetToken(userId = userId, expiresAt = createdAt.plusMinutes(10), createdAt = createdAt)
            .apply { this.attempts = attempts },
    )

    @Test
    fun `sumAttemptsSince counts only this account's rows inside the window`() {
        val now = OffsetDateTime.now()
        val owner = user()
        val other = user()
        val idle = user()
        row(owner.id, 5, now.minusHours(1))
        row(owner.id, 4, now.minusHours(2))
        row(owner.id, 2, now.minusHours(30))
        row(other.id, 5, now.minusHours(1))

        assertEquals(9, tokens.sumAttemptsSince(owner.id, now.minusHours(24)))
        assertEquals(0, tokens.sumAttemptsSince(idle.id, now.minusHours(24)))
    }
}
