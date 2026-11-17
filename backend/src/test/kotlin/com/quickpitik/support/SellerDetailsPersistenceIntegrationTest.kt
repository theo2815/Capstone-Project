package com.quickpitik.support

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.LivenessSession
import com.quickpitik.entity.RepeatMatch
import com.quickpitik.entity.RepeatMatchKind
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerReasonCode
import com.quickpitik.repository.ID_NUMBER_LOCK_KEY
import com.quickpitik.repository.PAYOUT_NUMBER_LOCK_KEY
import com.quickpitik.repository.PHONE_KEY_LOCK_KEY
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.service.liveness.LivenessSessionService
import jakarta.persistence.EntityManager
import java.math.BigDecimal
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.sql.DataSource
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** V93: the seller-details table, its two checks, the no-cascade rule and the upload gate query. */
class SellerDetailsPersistenceIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var repository: SellerDetailsRepository
    @Autowired private lateinit var entityManager: EntityManager
    @Autowired private lateinit var livenessSessions: LivenessSessionService
    @Autowired private lateinit var payoutRepository: PayoutAccountRepository
    @Autowired private lateinit var dataSource: DataSource
    @Autowired private lateinit var transactionManager: PlatformTransactionManager

    private fun photographer(): UUID = jdbc.queryForObject(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Seller Test', 'PHOTOGRAPHER') RETURNING id",
        UUID::class.java, "seller-${UUID.randomUUID()}@test.local",
    )!!

    private fun insert(
        userId: UUID,
        idType: String = "PASSPORT",
        idBack: String? = null,
        status: String = "MANUAL_REVIEW",
        reason: String? = null,
    ) = jdbc.update(
        "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, id_back_key, selfie_key, " +
            "address, phone, phone_key, status, rejection_reason, revision, submitted_at) " +
            "VALUES (?, 'Juan Dela Cruz', ?, 'seller-details/x/f.jpg', ?, 'seller-details/x/s.jpg', 'Cebu', '09171234567', " +
            "'9171234567', ?, ?, 1, now())",
        userId, idType, idBack, status, reason,
    )

    @Test
    fun `V93 applies and the entity maps the table`() {
        val user = photographer()
        insert(user)
        val row = repository.findById(user).orElseThrow()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf("seller-details/x/f.jpg", "seller-details/x/s.jpg"), row.fileKeys())
        assertEquals("9171234567", row.phoneKey)
    }

    @Test
    fun `reason codes and repeat matches round-trip as jsonb`() {
        val user = photographer()
        val other = photographer()
        insert(user)
        val row = repository.findById(user).orElseThrow()
        assertEquals(emptyList(), row.reasonCodes)
        assertEquals(emptyList(), row.repeatMatches)

        row.reasonCodes = listOf(SellerReasonCode.LIVENESS_LOW, SellerReasonCode.FACE_MISMATCH)
        row.repeatMatches = listOf(RepeatMatch(other, listOf(RepeatMatchKind.PHONE, RepeatMatchKind.ID_NUMBER)))
        row.livenessConfidence = BigDecimal("87.25")
        row.livenessReferenceKey = "seller-details/x/ref.jpg"
        repository.saveAndFlush(row)

        val again = repository.findById(user).orElseThrow()
        assertEquals(listOf(SellerReasonCode.LIVENESS_LOW, SellerReasonCode.FACE_MISMATCH), again.reasonCodes)
        assertEquals(listOf(RepeatMatch(other, listOf(RepeatMatchKind.PHONE, RepeatMatchKind.ID_NUMBER))), again.repeatMatches)
        assertEquals(0, BigDecimal("87.25").compareTo(again.livenessConfidence))
        assertTrue("seller-details/x/ref.jpg" in again.fileKeys(), "the reference image is deleted with the other files")
        assertEquals("[\"LIVENESS_LOW\", \"FACE_MISMATCH\"]",
            jdbc.queryForObject("SELECT reason_codes::text FROM photographer_seller_details WHERE user_id = ?", String::class.java, user))
    }

    @Test
    fun `a liveness session needs an event exactly when its purpose is a runner removal`() {
        val user = photographer()
        val event = jdbc.queryForObject(
            "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Liveness run', DATE '2026-12-01', 'Cebu', 'ACTIVE') RETURNING id",
            UUID::class.java, "liveness-${UUID.randomUUID()}",
        )!!
        fun session(purpose: String, eventId: UUID?, consent: String? = "1.3") = jdbc.update(
            "INSERT INTO liveness_sessions (session_id, user_id, purpose, event_id, consent_version, created_at) VALUES (?, ?, ?, ?, ?, now())",
            "s-${UUID.randomUUID()}", user, purpose, eventId, consent,
        )
        assertThrows<DataIntegrityViolationException> { session("PHOTOGRAPHER_IDENTITY", event) }
        assertThrows<DataIntegrityViolationException> { session("RUNNER_PHOTO_REMOVAL", null) }
        assertThrows<DataIntegrityViolationException> { session("SOMETHING_ELSE", null) }
        session("PHOTOGRAPHER_IDENTITY", null)
        session("RUNNER_PHOTO_REMOVAL", event)

        val saved = entityManager.createQuery("SELECT s FROM LivenessSession s WHERE s.userId = :u", LivenessSession::class.java)
            .setParameter("u", user).resultList
        assertEquals(setOf(LivenessPurpose.PHOTOGRAPHER_IDENTITY, LivenessPurpose.RUNNER_PHOTO_REMOVAL), saved.map { it.purpose }.toSet())
        assertEquals(event, saved.single { it.purpose == LivenessPurpose.RUNNER_PHOTO_REMOVAL }.eventId)
        assertTrue(saved.all { it.consentVersion == "1.3" && it.consumedAt == null })
        // The account's sessions go with the account (ON DELETE CASCADE).
        jdbc.update("DELETE FROM liveness_sessions WHERE user_id = ?", user)
    }

    private fun livenessRow(user: UUID, purpose: String = "PHOTOGRAPHER_IDENTITY", eventId: UUID? = null): String {
        val id = "s-${UUID.randomUUID()}"
        jdbc.update(
            "INSERT INTO liveness_sessions (session_id, user_id, purpose, event_id, consent_version, created_at) VALUES (?, ?, ?, ?, '1.2', now())",
            id, user, purpose, eventId,
        )
        return id
    }

    private fun consumed(sessionId: String): Boolean =
        jdbc.queryForObject("SELECT consumed_at IS NOT NULL FROM liveness_sessions WHERE session_id = ?", Boolean::class.java, sessionId)!!

    @Test
    fun `a liveness session is claimed once`() {
        val user = photographer()
        val id = livenessRow(user)
        assertTrue(livenessSessions.claim(user, id, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null))
        assertTrue(consumed(id))
        assertFalse(livenessSessions.claim(user, id, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null), "a second submit finds it used")
        assertFalse(livenessSessions.claim(user, "s-unknown-${UUID.randomUUID()}", LivenessPurpose.PHOTOGRAPHER_IDENTITY, null))
    }

    @Test
    fun `a claim with another purpose or event changes nothing`() {
        val user = photographer()
        val other = photographer()
        val event = jdbc.queryForObject(
            "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Claim run', DATE '2026-12-01', 'Cebu', 'ACTIVE') RETURNING id",
            UUID::class.java, "claim-${UUID.randomUUID()}",
        )!!
        val photographerSession = livenessRow(user)
        val runnerSession = livenessRow(user, "RUNNER_PHOTO_REMOVAL", event)

        assertFalse(livenessSessions.claim(other, photographerSession, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null), "another account's")
        assertFalse(livenessSessions.claim(user, photographerSession, LivenessPurpose.RUNNER_PHOTO_REMOVAL, event), "another purpose")
        assertFalse(livenessSessions.claim(user, runnerSession, LivenessPurpose.RUNNER_PHOTO_REMOVAL, UUID.randomUUID()), "another event")
        assertFalse(livenessSessions.claim(user, runnerSession, LivenessPurpose.RUNNER_PHOTO_REMOVAL, null), "no event")
        assertFalse(consumed(photographerSession))
        assertFalse(consumed(runnerSession))

        // The matching event (IS NOT DISTINCT FROM) and the matching null both claim.
        assertTrue(livenessSessions.claim(user, runnerSession, LivenessPurpose.RUNNER_PHOTO_REMOVAL, event))
        assertTrue(livenessSessions.claim(user, photographerSession, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null))
        jdbc.update("DELETE FROM liveness_sessions WHERE user_id = ?", user)
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
    fun `the upload gate query checks both sets on real rows`() {
        val submitted = photographer().also { insert(it) }
        val auto = photographer().also { insert(it, status = "AUTO_APPROVED") }
        val approved = photographer().also { insert(it, status = "APPROVED") }
        val rejected = photographer().also { insert(it, status = "REJECTED", reason = "blurry") }
        val none = photographer()
        val gate = { id: UUID, on: Boolean -> repository.existsByUserIdAndStatusIn(id, SellerDetailsStatus.uploadGateSet(on)) }
        // Gate off (legacy): anything but REJECTED.
        assertTrue(gate(submitted, false))
        assertTrue(gate(auto, false))
        assertTrue(gate(approved, false))
        assertFalse(gate(rejected, false))
        assertFalse(gate(none, false))
        // Gate on (strict): AUTO_APPROVED or APPROVED only.
        assertFalse(gate(submitted, true))
        assertTrue(gate(auto, true))
        assertTrue(gate(approved, true))
        assertFalse(gate(rejected, true))
        assertFalse(gate(none, true))
    }

    // AA-I5: the three repeat-account lookups, each leaving out the caller.
    @Test
    fun `id number hash lookup finds accounts under any of the hashes and leaves out the caller`() {
        val (hmac, otherTypeHmac) = List(2) { UUID.randomUUID().toString().replace("-", "") }
        val (me, twin, otherType, stranger) = List(4) { photographer().also(::insert) }
        jdbc.update("UPDATE photographer_seller_details SET id_number_hmac = ? WHERE user_id IN (?, ?)", hmac, me, twin)
        // R-1: the same number stored under another ID type's hash.
        jdbc.update("UPDATE photographer_seller_details SET id_number_hmac = ? WHERE user_id = ?", otherTypeHmac, otherType)
        jdbc.update("UPDATE photographer_seller_details SET id_number_hmac = ? WHERE user_id = ?", "x$hmac".take(64), stranger)

        assertEquals(setOf(twin, otherType), repository.findUserIdsByIdNumberHmacIn(listOf(hmac, otherTypeHmac), me).toSet())
    }

    // R-9: the second submit of a number waits for the first to commit, then its lookup sees the first.
    @Test
    fun `a second transaction waits for the id number lock until the first ends`() {
        val hmac = UUID.randomUUID().toString().replace("-", "")
        val (first, second) = List(2) { photographer().also(::insert) }
        val pool = Executors.newSingleThreadExecutor()
        val pause = dataSource.connection.apply { autoCommit = false }
        try {
            pause.prepareStatement("SELECT pg_advisory_xact_lock($ID_NUMBER_LOCK_KEY, hashtext(?))").use { it.setString(1, hmac); it.execute() }
            pause.prepareStatement("UPDATE photographer_seller_details SET id_number_hmac = ? WHERE user_id = ?").use {
                it.setString(1, hmac); it.setObject(2, first); it.executeUpdate()
            }
            val lookup = pool.submit<List<UUID>> {
                TransactionTemplate(transactionManager).execute {
                    repository.lockIdNumber(hmac)
                    repository.findUserIdsByIdNumberHmacIn(listOf(hmac), second)
                }
            }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
            while (waitingOnIdNumberLock() < 1) {
                check(System.nanoTime() < deadline) { "the second transaction never queued on the id number lock" }
                Thread.sleep(20)
            }
            assertFalse(lookup.isDone)
            pause.commit() // ends the first transaction: the lock goes and its row is visible

            assertEquals(listOf(first), lookup.get(30, TimeUnit.SECONDS))
        } finally {
            pause.close()
            pool.shutdownNow()
        }
    }

    private fun waitingOnIdNumberLock(): Int = waitingOn(ID_NUMBER_LOCK_KEY)

    private fun waitingOn(namespace: Int): Int = jdbc.queryForObject(
        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objsubid = 2 AND classid::bigint = ? AND NOT granted",
        Int::class.java, namespace,
    )!!

    // The same-moment gap of R-9 for the phone and the payout number: a second holder of the same value waits.
    @Test
    fun `the phone key and payout number locks make a second transaction wait until the first ends`() {
        assertSecondWaits(PHONE_KEY_LOCK_KEY) { repository.lockPhoneKey(it) }
        assertSecondWaits(PAYOUT_NUMBER_LOCK_KEY) { payoutRepository.lockAccountNumber(it) }
    }

    private fun assertSecondWaits(namespace: Int, lock: (String) -> Unit) {
        val value = UUID.randomUUID().toString()
        val pool = Executors.newSingleThreadExecutor()
        val first = dataSource.connection.apply { autoCommit = false }
        try {
            first.prepareStatement("SELECT pg_advisory_xact_lock($namespace, hashtext(?))").use { it.setString(1, value); it.execute() }
            val second = pool.submit { TransactionTemplate(transactionManager).execute { lock(value) } }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
            while (waitingOn(namespace) < 1) {
                check(System.nanoTime() < deadline) { "the second transaction never queued on lock $namespace" }
                Thread.sleep(20)
            }
            assertFalse(second.isDone)
            first.commit() // the first transaction ends, so the second gets the lock
            second.get(30, TimeUnit.SECONDS)
        } finally {
            first.close()
            pool.shutdownNow()
        }
    }

    @Test
    fun `phone key lookup finds the other accounts and leaves out the caller`() {
        val key = (1_000_000_000L + (Math.random() * 8_999_999_999L).toLong()).toString()
        val (me, twin) = List(2) { photographer().also(::insert) }
        jdbc.update("UPDATE photographer_seller_details SET phone_key = ? WHERE user_id IN (?, ?)", key, me, twin)

        assertEquals(listOf(twin), repository.findUserIdsByPhoneKey(key, me))
    }

    @Test
    fun `payout number lookup finds each other account once and leaves out the caller`() {
        val number = "09" + (100_000_000L + (Math.random() * 899_999_999L).toLong())
        val (me, twin, other) = List(3) { photographer() }
        fun account(user: UUID, accountNumber: String) = jdbc.update(
            "INSERT INTO photographer_payout_accounts (id, user_id, method, account_number, account_name) VALUES (?, ?, 'GCASH', ?, 'X')",
            UUID.randomUUID(), user, accountNumber,
        )
        account(me, number)
        account(twin, number)
        account(twin, number) // two rows on one account still name it once
        account(other, "09000000000")

        assertEquals(listOf(twin), payoutRepository.findUserIdsByAccountNumbers(listOf(number, "09111111111"), me))
    }

    @Test
    fun `queue lists manual review first, oldest first, and filters by status`() {
        fun record(status: String, submittedAt: String, reason: String? = null) = photographer().also {
            jdbc.update(
                "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, address, phone, " +
                    "phone_key, status, rejection_reason, revision, submitted_at) " +
                    "VALUES (?, 'Q', 'PASSPORT', 'k/f', 'k/s', 'Cebu', '09171234567', '9171234567', ?, ?, 1, ?::timestamptz)",
                it, status, reason, submittedAt,
            )
        }
        val approvedOld = record("APPROVED", "2020-01-01T00:00:00Z")
        val submittedNew = record("MANUAL_REVIEW", "2020-01-03T00:00:00Z")
        val submittedOld = record("MANUAL_REVIEW", "2020-01-02T00:00:00Z")
        val rejected = record("REJECTED", "2020-01-04T00:00:00Z", "blurry")
        val auto = record("AUTO_APPROVED", "2019-12-31T00:00:00Z")
        val mine = setOf(approvedOld, submittedNew, submittedOld, rejected, auto)
        val all = repository.findForAdmin(null, OffsetLimitPageable(PaginationParams.of(0, 200))).content.map { it.userId }.filter { it in mine }
        assertEquals(listOf(submittedOld, submittedNew), all.take(2))
        assertEquals(setOf(approvedOld, rejected, auto), all.drop(2).toSet())
        assertTrue(all.indexOf(approvedOld) < all.indexOf(rejected), "non-submitted rows stay oldest first")

        val onlyRejected = repository.findForAdmin(SellerDetailsStatus.REJECTED, OffsetLimitPageable(PaginationParams.of(0, 200)))
        assertTrue(onlyRejected.content.all { it.status == SellerDetailsStatus.REJECTED })
        assertTrue(rejected in onlyRejected.content.map { it.userId })
        assertEquals(onlyRejected.content.size.toLong(), onlyRejected.totalElements)
    }
}
