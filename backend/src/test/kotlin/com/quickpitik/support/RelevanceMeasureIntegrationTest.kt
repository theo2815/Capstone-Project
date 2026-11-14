package com.quickpitik.support

import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.RelevanceCaseRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-1.6: the measuring and sampling-stage queries against real Postgres (plan 5.8). The container is shared, so
// fixtures sit in 2003 and assertions look only at this class's own rows.
class RelevanceMeasureIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `saving a measurement keeps the words mark`() {
        val b = batch()
        val token = claim(b)
        val counted = photo(b)
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, words_counted_at) VALUES (?, ?, ?)", counted, b.id, WORDS_AT)
        val fresh = photo(b)

        assertEquals(1, save(counted, b, token, fg = 50.0))
        assertEquals(1, save(fresh, b, token, fg = null))

        val kept = check(counted)
        assertEquals(WORDS_AT.toInstant(), (kept["words_counted_at"] as java.sql.Timestamp).toInstant())
        assertEquals(NOW.toInstant(), (kept["measured_at"] as java.sql.Timestamp).toInstant())
        assertEquals(listOf(50.0, 60.0, 40.0, 30.0, 0.9, 0.0),
            listOf("fg_sharpness", "fg_brightness", "sharpness", "brightness", "person_conf", "race_conf").map { kept[it] })
        val new = check(fresh)
        assertNull(new["words_counted_at"])
        assertNull(new["fg_sharpness"], "no foreground stays null")
        assertEquals(0.0, new["race_conf"], "an absent label is 0, never null")
    }

    @Test
    fun `a measurement written with a stale claim changes nothing`() {
        val b = batch()
        val live = claim(b)
        val p = photo(b)
        val stale = UUID.randomUUID()

        assertEquals(0, save(p, b, stale))
        assertEquals(0, tx.execute { batches.saveMeasureError(p, b.id, stale, "BAD_IMAGE") })
        assertEquals(0, tx.execute { batches.saveMeasureRetry(p, b.id, stale, 5) })
        assertEquals(0, rows(p), "a lost claim writes no row")

        tx.execute { batches.releaseCheck(b.id, live) }
        assertEquals(0, save(p, b, live), "a released claim writes nothing either")
        assertEquals(0, rows(p))
    }

    @Test
    fun `the next photos to measure skip measured and failed ones`() {
        val b = batch()
        val token = claim(b)
        // Postgres orders uuids byte by byte, which is their text order (Java's compareTo is signed).
        val photos = List(5) { photo(b) }.sortedBy { it.toString() }
        val (measured, failed, retried, trashed, fresh) = photos
        // An earlier check measured one photo; this check samples all five.
        val earlier = case(b, "CLOSED")
        sample(earlier, listOf(measured))
        assertEquals(1, save(measured, b, token))
        tx.execute { batches.saveMeasureError(failed, b.id, token, "BAD_IMAGE") }
        tx.execute { batches.saveMeasureRetry(retried, b.id, token, 5) }
        jdbc.update("UPDATE photos SET deleted_at = ? WHERE id = ?", NOW, trashed)
        val current = case(b, "SAMPLING")
        sample(current, photos)

        val next = batches.nextToMeasure(current, 10)
        assertEquals(listOf(retried, fresh), next.map { it.photoId }, "photo id order")
        assertEquals("relmeasure/$fresh", next.last().s3Key)
        assertEquals(listOf(retried), batches.nextToMeasure(current, 1).map { it.photoId })
    }

    @Test
    fun `a fifth failed try gives the photo up`() {
        val b = batch()
        val token = claim(b)
        val p = photo(b)
        repeat(4) { assertEquals(1, tx.execute { batches.saveMeasureRetry(p, b.id, token, 5) }) }
        assertEquals(4, (check(p)["measure_tries"] as Number).toInt())
        assertNull(check(p)["measure_error"], "four tries: still to measure")

        tx.execute { batches.saveMeasureRetry(p, b.id, token, 5) }
        assertEquals("GAVE_UP", check(p)["measure_error"])
        assertEquals(5, (check(p)["measure_tries"] as Number).toInt())
    }

    @Test
    fun `the sample scores leave out trashed photos and show unmeasured ones empty`() {
        val b = batch()
        val token = claim(b)
        val (measured, failed, unmeasured, trashed) = List(4) { photo(b) }
        save(measured, b, token, fg = 12.0)
        save(trashed, b, token)
        tx.execute { batches.saveMeasureError(failed, b.id, token, "NO_FILE") }
        jdbc.update("UPDATE photos SET deleted_at = ? WHERE id = ?", NOW, trashed)
        val c = case(b, "SAMPLING")
        sample(c, listOf(measured, failed, unmeasured, trashed))

        val rows = batches.sampleScores(c)
        assertEquals(3, rows.size)
        assertEquals(listOf(12.0), rows.mapNotNull { it.fgSharpness })
        assertEquals(listOf("NO_FILE"), rows.mapNotNull { it.measureError })
        assertEquals(1, rows.count { it.personConf == null && it.measureError == null }, "the unmeasured photo")
    }

    @Test
    fun `sampling cases are listed only while no claim is live`() {
        val b = batch()
        val c = case(b, "SAMPLING")
        val decidingBatch = batch().also { case(it, "DECIDING") }

        fun listed() = cases.findSampling(NOW, 10_000).filter { it.batchId == b.id || it.batchId == decidingBatch.id }
        val row = listed().single()
        assertEquals(listOf(c, b.event, b.photographer), listOf(row.caseId, row.eventId, row.photographerId))

        val token = claim(b)
        assertTrue(listed().isEmpty(), "claimed by a copy")
        tx.execute { batches.releaseCheck(b.id, token) }
        assertEquals(1, listed().size)
    }

    @Test
    fun `the stage writes need the claim and the sampling stage`() {
        val b = batch()
        val c = case(b, "SAMPLING")
        val stale = UUID.randomUUID()
        val token = claim(b)

        assertEquals(0, tx.execute { cases.startRoundTwo(c, stale) })
        assertEquals(1, tx.execute { cases.startRoundTwo(c, token) })
        assertEquals(0, tx.execute { cases.startRoundTwo(c, token) }, "round two is taken once")
        assertEquals(0, tx.execute { cases.finishSampling(c, stale, 10, 9, 1, 1, 1, 2, "MIXED") })
        assertEquals(1, tx.execute { cases.finishSampling(c, token, 10, 9, 1, 1, 1, 2, "MIXED") })
        val done = jdbc.queryForMap("SELECT * FROM relevance_cases WHERE id = ?", c)
        assertEquals(listOf("DECIDING", 2, 10, 9, 1, 1, 1, 2, "MIXED"),
            listOf("stage", "sample_round", "sample_size", "measured", "blurry", "dark", "not_race", "junk", "band")
                .map { (done[it] as? Number)?.toInt() ?: done[it] })
        assertEquals(0, tx.execute { cases.closeSkipped(c, token, NOW) }, "a case past sampling is not closed here")

        val otherBatch = batch()
        val otherToken = claim(otherBatch)
        val other = case(otherBatch, "SAMPLING")
        assertEquals(0, tx.execute { cases.closeSkipped(other, stale, NOW) })
        assertEquals(1, tx.execute { cases.closeSkipped(other, otherToken, NOW) })
        val closed = jdbc.queryForMap("SELECT stage, close_reason FROM relevance_cases WHERE id = ?", other)
        assertEquals(mapOf("stage" to "CLOSED", "close_reason" to "SKIPPED"), closed)
    }

    @Test
    fun `the deciding stage writes need the claim and the deciding stage`() {
        val b = batch()
        val c = case(b, "DECIDING")
        fun listed(waitCutoff: OffsetDateTime) = cases.findDeciding(NOW, waitCutoff, 10_000).filter { it.batchId == b.id }
        // created_at is the database's now(), long after NOW: only the far wait cutoff lets the case through unscored.
        assertTrue(listed(NOW).isEmpty(), "version 1 has not scored the case's upload and the wait is not over")
        assertEquals(1, listed(NOW.plusYears(200)).size, "past the wait")
        jdbc.update("UPDATE batch_relevance SET scored_upload_at = ? WHERE id = ?", NOW, b.id)
        assertEquals(c, listed(NOW).single().caseId, "scored for the same upload")

        val token = claim(b)
        assertTrue(listed(NOW).isEmpty(), "claimed by a copy")
        val stale = UUID.randomUUID()
        assertEquals(0, tx.execute { cases.raiseDecideAttempts(c, stale) })
        assertEquals(1, tx.execute { cases.raiseDecideAttempts(c, token) })
        assertEquals(0, verdict(c, stale))
        assertEquals(1, verdict(c, token))
        val row = jdbc.queryForMap("SELECT * FROM relevance_cases WHERE id = ?", c)
        assertEquals(listOf("DECIDED", 1, "CANT_TELL", "CONTRADICTS", "JUNK_DUMP", "r", "junk_share", "{}", "m", "rv1", true, null, 7),
            listOf("stage", "decide_attempts", "verdict", "verdict_note", "model_verdict", "admin_reason", "evidence_used", "summary_sent",
                "model", "prompt_version", "copy_low", "date_low", "words_photos").map { (row[it] as? Short)?.toInt() ?: row[it] })
        assertEquals(NOW.toInstant(), (row["decided_at"] as java.sql.Timestamp).toInstant())
        assertEquals(0, verdict(c, token), "decided once")
        assertEquals(0, tx.execute { cases.raiseDecideAttempts(c, token) }, "no call once decided")

        val other = batch()
        val otherToken = claim(other)
        val skipped = case(other, "DECIDING")
        assertEquals(0, tx.execute { cases.closeSkippedDeciding(skipped, stale, NOW) })
        assertEquals(1, tx.execute { cases.closeSkippedDeciding(skipped, otherToken, NOW) })
        assertEquals(mapOf("stage" to "CLOSED", "close_reason" to "SKIPPED"),
            jdbc.queryForMap("SELECT stage, close_reason FROM relevance_cases WHERE id = ?", skipped))
        assertEquals(0, tx.execute { cases.closeSkippedDeciding(c, token, NOW) }, "a decided case is not closed here")

        // A failed call stores no model verdict, reason, evidence or rule result.
        val failed = case(other, "DECIDING")
        assertEquals(1, tx.execute {
            cases.saveVerdict(failed, otherToken, NOW, "CANT_TELL", "CALL_FAILED", null, null, null, "{}", "m", "rv1", null, null, 0)
        })
        assertEquals("CALL_FAILED", jdbc.queryForObject("SELECT verdict_note FROM relevance_cases WHERE id = ?", String::class.java, failed))
    }

    private fun verdict(c: UUID, token: UUID): Int = tx.execute {
        cases.saveVerdict(c, token, NOW, "CANT_TELL", "CONTRADICTS", "JUNK_DUMP", "r", "junk_share", "{}", "m", "rv1", true, null, 7)
    }!!

    @Test
    fun `a batch is still allowed only while its event is public and its photographer active`() {
        val ok = batch()
        assertTrue(batches.stillAllowed(ok.id))
        val unlisted = batch().also { jdbc.update("UPDATE events SET visibility = 'UNLISTED' WHERE id = ?", it.event) }
        // Only a race event can be hidden (ck_events_race_only).
        val hidden = batch().also { jdbc.update("UPDATE events SET event_type = 'ROAD_RACE', hidden_at = ? WHERE id = ?", NOW, it.event) }
        val deleted = batch().also { jdbc.update("UPDATE events SET deleted_at = ? WHERE id = ?", NOW, it.event) }
        val suspended = batch().also { jdbc.update("UPDATE users SET suspended_at = ? WHERE id = ?", NOW, it.photographer) }
        listOf(unlisted, hidden, deleted, suspended).forEach { assertFalse(batches.stillAllowed(it.id), it.toString()) }
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun batch(): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relmeasure-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2003-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relmeasure-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 40, ?, ?)",
            event, photographer, NOW, NOW,
        )
        tx.execute { batches.ensureBatch(event, photographer) }
        return Batch(batches.findByEventIdAndPhotographerId(event, photographer)!!.id, event, photographer)
    }

    private fun claim(b: Batch): UUID {
        val token = UUID.randomUUID()
        assertEquals(1, tx.execute { batches.claimCheck(b.id, token, NOW.plusMinutes(10), NOW) })
        return token
    }

    private fun photo(b: Batch): UUID {
        val id = UUID.randomUUID()
        jdbc.update(
            "INSERT INTO photos (id, event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, ?, 125, 'LIVE')",
            id, b.event, b.photographer, "relmeasure/$id",
        )
        return id
    }

    private fun case(b: Batch, stage: String): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) VALUES (?, 'QUIET', ?, 40, ?)",
        b.id, NOW, stage,
    )

    private fun sample(caseId: UUID, photos: List<UUID>) = photos.forEach {
        jdbc.update("INSERT INTO relevance_case_photos (case_id, photo_id, round) VALUES (?, ?, 1)", caseId, it)
    }

    private fun save(p: UUID, b: Batch, token: UUID, fg: Double? = 50.0): Int =
        tx.execute { batches.saveMeasurement(p, b.id, token, NOW, fg, 60.0, 40.0, 30.0, 0.9, 0.0) }!!

    private fun check(p: UUID): Map<String, Any?> = jdbc.queryForMap("SELECT * FROM relevance_photo_checks WHERE photo_id = ?", p)

    private fun rows(p: UUID): Int =
        jdbc.queryForObject("SELECT count(*) FROM relevance_photo_checks WHERE photo_id = ?", Int::class.java, p)!!

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2003-03-04T05:06:07.123456Z")
        val WORDS_AT: OffsetDateTime = NOW.minusHours(1)
    }
}
