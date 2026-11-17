package com.quickpitik.support

import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** AA-I22: who the identity notice goes to. */
class IdentityNoticeRepositoryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var users: UserRepository

    private val startsOn = LocalDate.of(2026, 12, 7)
    private val tag = UUID.randomUUID().toString().take(8)

    private fun user(name: String, role: String = "PHOTOGRAPHER", suspended: Boolean = false): UUID = jdbc.queryForObject(
        "INSERT INTO users (email, password_hash, name, role, suspended_at) VALUES (?, 'unused', ?, ?, " +
            (if (suspended) "now()" else "NULL") + ") RETURNING id",
        UUID::class.java, "$name-$tag@test.local", name, role,
    )!!

    private fun seller(userId: UUID, status: String) = jdbc.update(
        "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, " +
            "address, phone, phone_key, status, rejection_reason, revision, submitted_at) " +
            "VALUES (?, 'Juan Dela Cruz', 'PASSPORT', 'seller-details/x/f.jpg', 'seller-details/x/s.jpg', 'Cebu', " +
            "'09171234567', '9171234567', ?, ?, 1, now())",
        userId, status, if (status == "REJECTED") "blurry" else null,
    )

    @Test
    fun `recipients exclude passed, suspended, closed and runner accounts`() {
        user("none")
        seller(user("review"), "MANUAL_REVIEW")
        seller(user("rejected"), "REJECTED")
        seller(user("auto"), "AUTO_APPROVED")
        seller(user("approved"), "APPROVED")
        user("suspended", suspended = true)
        seller(user("closed", suspended = true), "MANUAL_REVIEW") // a closed photographer is anonymised and suspended
        user("runner", role = "RUNNER")
        user("admin", role = "ADMIN")

        val passed = SellerDetailsStatus.entries.filter { it.passesIdentity() }
        val names = users.findIdentityNoticeRecipients(passed, startsOn).filter { it.email.endsWith("-$tag@test.local") }.map { it.name }

        assertEquals(setOf("none", "review", "rejected"), names.toSet())
        assertEquals(3, names.size)
    }

    @Test
    fun `the claim marks a user once and the recipient query then leaves them out`() {
        val id = user("claimed")
        val passed = SellerDetailsStatus.entries.filter { it.passesIdentity() }
        fun listed(on: LocalDate) = users.findIdentityNoticeRecipients(passed, on).any { it.id == id }

        // no outer transaction here: a missing @Transactional on the claim fails in this test
        assertEquals(1, users.claimIdentityNotice(id, startsOn))
        assertEquals(0, users.claimIdentityNotice(id, startsOn))
        assertTrue(!listed(startsOn))
        assertTrue(listed(startsOn.plusDays(7)))

        assertEquals(1, users.claimIdentityNotice(id, startsOn.plusDays(7)))
        users.clearIdentityNotice(id, startsOn.plusDays(7))
        assertTrue(listed(startsOn.plusDays(7)))
    }
}
