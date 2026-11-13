package com.quickpitik.support

import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.RelevanceCaseRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-1.5: the version 2 check queries against real Postgres (plan 5.8). findCheckDue is global with a limit
// and the container is shared, so fixture uploads sit in 2002 (the ordering test in 1995) and assertions look
// only at this class's own batch keys.
class RelevanceCheckQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `a public batch quiet for ten minutes is due once per upload`() {
        val young = batchKey(lastUpload = QUIET_CUTOFF.plusNanos(1_000))
        assertNull(dueRow(young), "still uploading and under the count trigger")

        val b = batchKey()
        val first = assertNotNull(dueRow(b))
        assertEquals("QUIET", first.trigger)
        assertNull(first.batchId, "due before the batch has a row")
        assertEquals(UPLOAD.toInstant(), first.lastUploadAt, "microseconds survive the projection")

        checked(b, uploadAt = UPLOAD, photos = 40)
        assertNull(dueRow(b), "checked for its newest upload")

        val later = UPLOAD.plusMinutes(5)
        jdbc.update("UPDATE event_photographer SET last_upload_at = ? WHERE event_id = ?", later, b.event)
        assertEquals("QUIET", dueRow(b)?.trigger, "a newer upload makes it due again")
    }

    @Test
    fun `a batch still uploading is due when it grew by the count trigger`() {
        val b = batchKey(lastUpload = NOW.minusMinutes(1), photoCount = 499)
        assertNull(dueRow(b))

        jdbc.update("UPDATE event_photographer SET photo_count = 500 WHERE event_id = ?", b.event)
        val due = assertNotNull(dueRow(b))
        assertEquals("COUNT", due.trigger)
        assertEquals(500, due.photoCount)

        checked(b, uploadAt = NOW.minusMinutes(1), photos = 500)
        jdbc.update("UPDATE event_photographer SET photo_count = 999 WHERE event_id = ?", b.event)
        assertNull(dueRow(b), "499 more since the last check")
        jdbc.update("UPDATE event_photographer SET photo_count = 1000 WHERE event_id = ?", b.event)
        assertEquals("COUNT", dueRow(b)?.trigger)
    }

    @Test
    fun `a private event's batch is not due, and is due after the event turns public`() {
        val b = batchKey(visibility = "UNLISTED")
        assertNull(dueRow(b))
        jdbc.update("UPDATE events SET visibility = 'PUBLIC' WHERE id = ?", b.event)
        assertNotNull(dueRow(b))
    }

    @Test
    fun `batches of a hidden or deleted event and of a suspended photographer are not due`() {
        // Only a race event can be hidden (ck_events_race_only).
        val hidden = batchKey().also { jdbc.update("UPDATE events SET event_type = 'ROAD_RACE', hidden_at = ? WHERE id = ?", NOW, it.event) }
        val deleted = batchKey().also { jdbc.update("UPDATE events SET deleted_at = ? WHERE id = ?", NOW, it.event) }
        val suspended = batchKey().also { jdbc.update("UPDATE users SET suspended_at = ? WHERE id = ?", NOW, it.photographer) }
        listOf(hidden, deleted, suspended).forEach { assertNull(dueRow(it), it.toString()) }
    }

    @Test
    fun `a batch with a check in flight or a live claim is not due`() {
        val inFlight = listOf("SAMPLING", "DECIDING", "DECIDED").map { stage -> batchKey().also { case(batchId(it), stage) } }
        inFlight.forEach { assertNull(dueRow(it), it.toString()) }
        val applied = batchKey().also { case(batchId(it), "APPLIED") }
        assertNotNull(dueRow(applied), "an applied case is not in flight")

        val claimed = batchKey()
        val id = batchId(claimed)
        val token = UUID.randomUUID()
        assertEquals(1, claim(id, token, until = NOW.plusMinutes(10)))
        assertNull(dueRow(claimed))
        assertNotNull(dueFor(id, token), "the claim holder still reads it as due")
        assertNull(dueFor(id, UUID.randomUUID()), "another token does not")

        tx.execute { batches.releaseCheck(id, token) }
        assertNotNull(dueRow(claimed), "released")
        assertNull(dueFor(id, token), "a released token no longer reads it")
    }

    @Test
    fun `due batches come least recently served first, up to the limit`() {
        val upload = OffsetDateTime.parse("1995-01-01T00:00:00.000001Z")
        val neverA = batchKey(lastUpload = upload)
        val neverB = batchKey(lastUpload = upload.plusSeconds(1))
        val servedOld = batchKey(lastUpload = upload).also { served(it, upload.plusDays(1)) }
        val servedNew = batchKey(lastUpload = upload).also { served(it, upload.plusDays(2)) }
        val cutoff = upload.plusDays(3)

        fun page(limit: Int) = batches.findCheckDue(cutoff, cutoff, 1_000_000, limit).map { Key(it.eventId, it.photographerId) }

        assertEquals(listOf(neverA, neverB, servedOld), page(3))
        assertEquals(listOf(neverA, neverB, servedOld, servedNew), page(10))
    }

    @Test
    fun `a second claim on a live claim changes nothing`() {
        val id = batchId(batchKey())
        val first = UUID.randomUUID()
        assertEquals(1, claim(id, first, until = NOW.plusMinutes(10)))
        assertEquals(0, claim(id, UUID.randomUUID(), until = NOW.plusMinutes(10), now = NOW.plusMinutes(9)))
        assertEquals(first, jdbc.queryForObject("SELECT check_claim FROM batch_relevance WHERE id = ?", UUID::class.java, id))
        assertEquals(0, tx.execute { batches.releaseCheck(id, UUID.randomUUID()) }, "only the holder releases")
    }

    @Test
    fun `a claim past its time can be taken`() {
        val b = batchKey()
        val id = batchId(b)
        assertEquals(1, claim(id, UUID.randomUUID(), until = NOW.plusMinutes(10)))
        val second = UUID.randomUUID()
        assertEquals(1, claim(id, second, until = NOW.plusMinutes(21), now = NOW.plusMinutes(11)))
        assertEquals(second, jdbc.queryForObject("SELECT check_claim FROM batch_relevance WHERE id = ?", UUID::class.java, id))
        assertEquals(0, tx.execute { batches.markChecked(id, UUID.randomUUID(), UPLOAD, 40) }, "a lost claim writes nothing")
        assertEquals(1, tx.execute { batches.markChecked(id, second, UPLOAD, 40) })
    }

    @Test
    fun `the sample is distinct photos, includes hidden ones and leaves out trashed and removed ones`() {
        val b = batchKey()
        val kept = listOf("LIVE", "LIVE", "HIDDEN", "PROCESSING").map { photo(b, status = it) }
        photo(b, trashed = true)
        photo(b, removal = "REVIEW")
        photo(b, removal = "REMOVED")
        val other = batchKey()
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id) VALUES (?, ?)", b.event, other.photographer)
        photo(Key(b.event, other.photographer))
        val caseId = case(batchId(b), "SAMPLING")

        assertEquals(4, sample(caseId, b, round = 1, size = 300))
        assertEquals(kept.toSet(), sampled(caseId).keys)
        assertEquals(4, cases.countSample(caseId))
        assertEquals(0, sample(caseId, b, round = 1, size = 300), "a photo is sampled once per case")
    }

    @Test
    fun `a batch smaller than the sample is sampled whole`() {
        val small = batchKey()
        val photos = List(5) { photo(small) }
        val smallCase = case(batchId(small), "SAMPLING")
        assertEquals(5, sample(smallCase, small, round = 1, size = 300))
        assertEquals(photos.toSet(), sampled(smallCase).keys)

        val big = batchKey()
        repeat(10) { photo(big) }
        val bigCase = case(batchId(big), "SAMPLING")
        assertEquals(3, sample(bigCase, big, round = 1, size = 3), "a larger batch gets the sample size")
    }

    @Test
    fun `round two picks only photos not in round one`() {
        val b = batchKey()
        repeat(10) { photo(b) }
        val caseId = case(batchId(b), "SAMPLING")
        assertEquals(6, sample(caseId, b, round = 1, size = 6))
        assertEquals(4, sample(caseId, b, round = 2, size = 6))
        val rounds = sampled(caseId)
        assertEquals(10, rounds.size)
        assertEquals(6, rounds.values.count { it == 1 })
        assertEquals(4, rounds.values.count { it == 2 })
    }

    private data class Key(val event: UUID, val photographer: UUID)

    private fun batchKey(
        lastUpload: OffsetDateTime? = UPLOAD,
        photoCount: Int = 40,
        visibility: String = "PUBLIC",
    ): Key {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relcheck-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2002-03-04', 'Cebu', 'ACTIVE', ?)",
            "relcheck-${UUID.randomUUID()}", visibility,
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?, ?)",
            event, photographer, photoCount, lastUpload, lastUpload,
        )
        return Key(event, photographer)
    }

    private fun batchId(b: Key): UUID {
        tx.execute { batches.ensureBatch(b.event, b.photographer) }
        return batches.findByEventIdAndPhotographerId(b.event, b.photographer)!!.id
    }

    private fun claim(id: UUID, token: UUID, until: OffsetDateTime, now: OffsetDateTime = NOW): Int =
        tx.execute { batches.claimCheck(id, token, until, now) }!!

    // What a started check leaves on the batch: claimed, marked checked, released.
    private fun checked(b: Key, uploadAt: OffsetDateTime, photos: Int) {
        val id = batchId(b)
        val token = UUID.randomUUID()
        assertEquals(1, claim(id, token, until = NOW.plusMinutes(10)))
        assertEquals(1, tx.execute { batches.markChecked(id, token, uploadAt, photos) })
        assertEquals(1, tx.execute { batches.releaseCheck(id, token) })
    }

    private fun served(b: Key, at: OffsetDateTime) = jdbc.update(
        "INSERT INTO batch_relevance (event_id, photographer_id, check_served_at) VALUES (?, ?, ?)", b.event, b.photographer, at,
    )

    private fun case(batchId: UUID, stage: String): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) VALUES (?, 'QUIET', ?, 40, ?)",
        batchId, UPLOAD, stage,
    )

    private fun photo(b: Key, status: String = "LIVE", trashed: Boolean = false, removal: String = "NONE"): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status, deleted_at, removal_state) VALUES (?, ?, ?, 125, ?, ?, ?)",
        b.event, b.photographer, "relcheck/${UUID.randomUUID()}", status, if (trashed) NOW else null, removal,
    )

    private fun sample(caseId: UUID, b: Key, round: Int, size: Int): Int =
        tx.execute { cases.pickSample(caseId, b.event, b.photographer, round, size) }!!

    private fun sampled(caseId: UUID): Map<UUID, Int> =
        jdbc.queryForList("SELECT photo_id, round FROM relevance_case_photos WHERE case_id = ?", caseId)
            .associate { it["photo_id"] as UUID to (it["round"] as Number).toInt() }

    private fun dueRow(b: Key): CheckDue? =
        batches.findCheckDue(QUIET_CUTOFF, NOW, COUNT_TRIGGER, 10_000).singleOrNull { it.eventId == b.event && it.photographerId == b.photographer }

    private fun dueFor(id: UUID, token: UUID): CheckDue? =
        tx.execute { batches.findCheckDueFor(id, token, QUIET_CUTOFF, COUNT_TRIGGER) }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        // Non-zero microseconds: the quiet rule compares checked_upload_at with last_upload_at for equality.
        val NOW: OffsetDateTime = OffsetDateTime.parse("2002-03-04T05:06:07.123456Z")
        val QUIET_CUTOFF: OffsetDateTime = NOW.minusMinutes(10)
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
        const val COUNT_TRIGGER = 500
    }
}
