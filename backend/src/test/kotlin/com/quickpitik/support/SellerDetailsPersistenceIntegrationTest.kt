package com.quickpitik.support

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.repository.SellerDetailsRepository
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.jdbc.core.JdbcTemplate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** V93: the seller-details table, its two checks, the no-cascade rule and the upload gate query. */
class SellerDetailsPersistenceIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var repository: SellerDetailsRepository

    private fun photographer(): UUID = jdbc.queryForObject(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Seller Test', 'PHOTOGRAPHER') RETURNING id",
        UUID::class.java, "seller-${UUID.randomUUID()}@test.local",
    )!!

    private fun insert(
        userId: UUID,
        idType: String = "PASSPORT",
        idBack: String? = null,
        status: String = "SUBMITTED",
        reason: String? = null,
    ) = jdbc.update(
        "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, id_back_key, selfie_key, " +
            "address, phone, status, rejection_reason, revision, submitted_at) " +
            "VALUES (?, 'Juan Dela Cruz', ?, 'seller-details/x/f.jpg', ?, 'seller-details/x/s.jpg', 'Cebu', '09171234567', ?, ?, 1, now())",
        userId, idType, idBack, status, reason,
    )

    @Test
    fun `V93 applies and the entity maps the table`() {
        val user = photographer()
        insert(user)
        val row = repository.findById(user).orElseThrow()
        assertEquals(SellerDetailsStatus.SUBMITTED, row.status)
        assertEquals(listOf("seller-details/x/f.jpg", "seller-details/x/s.jpg"), row.fileKeys())
    }

    @Test
    fun `a PhilSys record without a back and a rejection without a reason are refused`() {
        assertThrows<DataIntegrityViolationException> { insert(photographer(), idType = "PHILSYS") }
        assertThrows<DataIntegrityViolationException> { insert(photographer(), status = "REJECTED") }
        assertThrows<DataIntegrityViolationException> { insert(photographer(), idType = "OTHER") }
        insert(photographer(), idType = "PHILSYS", idBack = "seller-details/x/b.jpg")
        insert(photographer(), status = "REJECTED", reason = "blurry")
    }

    @Test
    fun `deleting a user who has a record fails instead of removing the record`() {
        val user = photographer()
        insert(user)
        assertThrows<DataIntegrityViolationException> { jdbc.update("DELETE FROM users WHERE id = ?", user) }
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photographer_seller_details WHERE user_id = ?", Int::class.java, user))
    }

    @Test
    fun `the upload gate query passes SUBMITTED and APPROVED and refuses REJECTED and no record`() {
        val submitted = photographer().also { insert(it) }
        val approved = photographer().also { insert(it, status = "APPROVED") }
        val rejected = photographer().also { insert(it, status = "REJECTED", reason = "blurry") }
        val none = photographer()
        val gate = { id: UUID -> repository.existsByUserIdAndStatusNot(id, SellerDetailsStatus.REJECTED) }
        assertTrue(gate(submitted))
        assertTrue(gate(approved))
        assertFalse(gate(rejected))
        assertFalse(gate(none))
    }

    @Test
    fun `the admin queue lists submitted records first, oldest first, and filters by status`() {
        fun record(status: String, submittedAt: String, reason: String? = null) = photographer().also {
            jdbc.update(
                "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, address, phone, " +
                    "status, rejection_reason, revision, submitted_at) VALUES (?, 'Q', 'PASSPORT', 'k/f', 'k/s', 'Cebu', '09171234567', ?, ?, 1, ?::timestamptz)",
                it, status, reason, submittedAt,
            )
        }
        val approvedOld = record("APPROVED", "2020-01-01T00:00:00Z")
        val submittedNew = record("SUBMITTED", "2020-01-03T00:00:00Z")
        val submittedOld = record("SUBMITTED", "2020-01-02T00:00:00Z")
        val rejected = record("REJECTED", "2020-01-04T00:00:00Z", "blurry")
        val mine = setOf(approvedOld, submittedNew, submittedOld, rejected)
        val all = repository.findForAdmin(null, OffsetLimitPageable(PaginationParams.of(0, 200))).content.map { it.userId }.filter { it in mine }
        assertEquals(listOf(submittedOld, submittedNew), all.take(2))
        assertEquals(setOf(approvedOld, rejected), all.drop(2).toSet())
        assertTrue(all.indexOf(approvedOld) < all.indexOf(rejected), "non-submitted rows stay oldest first")

        val onlyRejected = repository.findForAdmin(SellerDetailsStatus.REJECTED, OffsetLimitPageable(PaginationParams.of(0, 200)))
        assertTrue(onlyRejected.content.all { it.status == SellerDetailsStatus.REJECTED })
        assertTrue(rejected in onlyRejected.content.map { it.userId })
        assertEquals(onlyRejected.content.size.toLong(), onlyRejected.totalElements)
    }
}
