package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

// RV-2.3b (plan part 2, A.2): photo-by-photo hides for target HIDE_JUNK, and the restore step for a lowered target
// (HIDE_JUNK or NONE). The container is shared, so fixtures sit in 2008 and assertions look only at this class's rows.
class RelevanceJunkHoldsIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch()
        case(b, "HIDE_JUNK")
        val junk = photo(b, JUNK)
        val held = photo(b, GOOD, status = "HIDDEN").also { hold(b, it) }
        val noneBatch = batch()
        case(noneBatch, "NONE")
        val alsoHeld = photo(noneBatch, GOOD, status = "HIDDEN").also { hold(noneBatch, it) }

        service(mode = RelevanceMode.OFF).tick()

        assertEquals("HIDDEN", status(alsoHeld))

        assertEquals("LIVE", status(junk))
        assertEquals("HIDDEN", status(held))
        assertEquals(1L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch()
        case(b, "HIDE_JUNK")
        photo(b, JUNK)
        photo(b, GOOD, status = "HIDDEN").also { hold(b, it) }
        val before = untouched()

        service(mode = RelevanceMode.RECORD_ONLY).tick()

        assertEquals(before, untouched())
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val junkBatch = batch()
        case(junkBatch, "HIDE_JUNK", mode = "RECORD_ONLY")
        val junk = photo(junkBatch, JUNK)
        val noneBatch = batch()
        case(noneBatch, "NONE", mode = "RECORD_ONLY")
        val held = photo(noneBatch, GOOD, status = "HIDDEN").also { hold(noneBatch, it) }

        service().reconcileHolds()

        assertEquals("LIVE", status(junk))
        assertEquals(0L, holds.countLiveHolds(junkBatch.id, "AUTO"))
        assertEquals("HIDDEN", status(held))
        assertEquals(1L, holds.countLiveHolds(noneBatch.id, "AUTO"))
    }

    @Test
    fun `a mixed batch's junk photos are hidden as they are measured and the good ones stay live`() {
        val b = batch()
        case(b, "HIDE_JUNK")
        val blurry = photo(b, JUNK)
        val dark = photo(b, DARK)
        val notRace = photo(b, NOT_RACE)
        val good = photo(b, GOOD)
        val later = photo(b, null)
        val failed = photo(b, null).also { jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measure_error) VALUES (?, ?, 'GAVE_UP')", it, b.id) }
        val service = service()

        service.reconcileHolds()
        assertEquals(listOf("HIDDEN", "HIDDEN", "HIDDEN", "LIVE", "LIVE", "LIVE"),
            listOf(blurry, dark, notRace, good, later, failed).map(::status))

        measure(b, later, JUNK)
        service.reconcileHolds()

        assertEquals("HIDDEN", status(later))
        assertEquals("LIVE", status(good))
        assertEquals(4L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `a target lowered from all to junk restores only measured photos that are not junk`() {
        val b = batch()
        val c = case(b, "HIDE_ALL")
        val junk = photo(b, JUNK)
        val good = photo(b, GOOD)
        val unmeasured = photo(b, null)
        val wordsOnly = photo(b, null).also {
            jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, words_counted_at) VALUES (?, ?, ?)", it, b.id, NOW)
        }
        val service = service()
        service.reconcileHolds()
        assertEquals(listOf("HIDDEN"), listOf(junk, good, unmeasured, wordsOnly).map(::status).distinct())

        jdbc.update("UPDATE relevance_cases SET action = 'HIDE_JUNK' WHERE id = ?", c)
        service.reconcileHolds()

        assertEquals(listOf("HIDDEN", "LIVE", "HIDDEN", "HIDDEN"), listOf(junk, good, unmeasured, wordsOnly).map(::status))
        assertEquals("AUTO", jdbc.queryForObject("SELECT restored_by FROM flag_hidden_photos WHERE photo_id = ?", String::class.java, good))
        assertEquals(3L, holds.countLiveHolds(b.id, "AUTO"))
    }

    // G4: a photo that could not be measured is never junk, and the scan never tries it again.
    @Test
    fun `a photo that could not be measured comes back when the target drops from all to junk`() {
        val b = batch()
        val c = case(b, "HIDE_ALL")
        val failed = List(2) { photo(b, null) }
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measure_error) VALUES (?, ?, 'GAVE_UP')", failed[0], b.id)
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measure_error) VALUES (?, ?, 'BAD_IMAGE')", failed[1], b.id)
        val service = service()
        service.reconcileHolds()
        assertEquals(listOf("HIDDEN"), failed.map(::status).distinct())

        jdbc.update("UPDATE relevance_cases SET action = 'HIDE_JUNK' WHERE id = ?", c)
        service.reconcileHolds()

        assertEquals(listOf("LIVE"), failed.map(::status).distinct())
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
    }

    // Kotlin's isJunk answers "unknown" for a missing score; the junk hide does not hide it, so the restore brings it back.
    @Test
    fun `a measured photo whose junk rule is unknown comes back when the target drops from all to junk`() {
        val b = batch()
        val c = case(b, "HIDE_ALL")
        val unknown = photo(b, null).also {
            jdbc.update(
                "INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, sharpness, brightness, person_conf, race_conf) " +
                    "VALUES (?, ?, ?, 50, 50, NULL, 0.9, 0.9)", it, b.id, NOW,
            )
        }
        val service = service()
        service.reconcileHolds()
        assertEquals("HIDDEN", status(unknown))

        jdbc.update("UPDATE relevance_cases SET action = 'HIDE_JUNK' WHERE id = ?", c)
        service.reconcileHolds()

        assertEquals("LIVE", status(unknown))
    }

    @Test
    fun `a hold left under an older case of the batch is restored by the current case's lower target`() {
        val b = batch()
        val older = case(b, "HIDE_ALL", stage = "CLOSED")
        case(b, "HIDE_JUNK")
        val good = photo(b, GOOD, status = "HIDDEN").also { hold(b, it, caseId = older) }
        val junk = photo(b, JUNK, status = "HIDDEN").also { hold(b, it, caseId = older) }

        service().reconcileHolds()

        assertEquals("LIVE", status(good))
        assertEquals("HIDDEN", status(junk))
    }

    @Test
    fun `a target of none restores every automatic hold and leaves admin and photographer holds`() {
        val b = batch()
        val older = case(b, "HIDE_ALL", stage = "CLOSED")
        case(b, "NONE")
        val auto = List(3) { photo(b, JUNK, status = "HIDDEN").also { p -> hold(b, p, caseId = older) } }
        val byAdmin = photo(b, GOOD, status = "HIDDEN").also { hold(b, it, heldBy = "ADMIN") }
        val byPhotographer = photo(b, GOOD, status = "HIDDEN").also { hold(b, it, heldBy = "PHOTOGRAPHER") }
        // One chunk of two a tick: the step repeats on the next tick until no automatic hold is left.
        val service = service(chunkSize = 2, chunksPerTick = 1)

        service.reconcileHolds()
        assertEquals(1L, holds.countLiveHolds(b.id, "AUTO"))
        service.reconcileHolds()

        assertEquals(listOf("LIVE"), auto.map(::status).distinct())
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
        assertEquals(listOf("HIDDEN", "HIDDEN"), listOf(byAdmin, byPhotographer).map(::status))
        assertEquals(1L, holds.countLiveHolds(b.id, "ADMIN"))
        assertEquals(1L, holds.countLiveHolds(b.id, "PHOTOGRAPHER"))
    }

    @Test
    fun `a restored photo with no watermark goes back to processing`() {
        val b = batch()
        case(b, "NONE")
        val bare = photo(b, GOOD, status = "HIDDEN", watermark = null).also { hold(b, it) }
        val marked = photo(b, GOOD, status = "HIDDEN").also { hold(b, it) }

        service().reconcileHolds()

        assertEquals("PROCESSING", status(bare))
        assertEquals("LIVE", status(marked))
    }

    @Test
    fun `a photo another hold still keeps stays hidden`() {
        val b = batch()
        case(b, "NONE")
        val alsoAdmin = photo(b, GOOD, status = "HIDDEN").also { hold(b, it); hold(b, it, heldBy = "ADMIN") }
        val flagged = photo(b, GOOD, status = "HIDDEN").also {
            hold(b, it)
            jdbc.update("INSERT INTO flags (target_kind, target_id, reason, status) VALUES ('photo', ?, 'other', 'hidden')", it)
        }

        service().reconcileHolds()

        assertEquals(listOf("HIDDEN", "HIDDEN"), listOf(alsoAdmin, flagged).map(::status))
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `the event going private neither lifts nor blocks the junk hide`() {
        val b = batch()
        case(b, "HIDE_JUNK")
        val first = photo(b, JUNK)
        val service = service()
        service.reconcileHolds()
        jdbc.update("UPDATE events SET visibility = 'UNLISTED' WHERE id = ?", b.event)
        val second = photo(b, JUNK)

        service.reconcileHolds()

        assertEquals(listOf("HIDDEN", "HIDDEN"), listOf(first, second).map(::status))
        assertEquals(2L, holds.countLiveHolds(b.id, "AUTO"))
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    // fg sharpness, brightness, person and race confidence, against the cut-offs below (blur 20, dark 5, labels 0.6).
    private data class Scores(val fg: Double, val brightness: Double, val person: Double, val race: Double)

    private fun service(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, chunkSize: Int = 1000, chunksPerTick: Int = 10): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
            hide = RelevanceProperties.Hide(chunkSize = chunkSize, chunksPerTick = chunksPerTick))
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
            holds, photos, decisions, org.springframework.core.task.SyncTaskExecutor())
    }

    private fun batch(): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "reljunk-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Junk Run', DATE '2008-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "reljunk-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 0, ?, ?)",
            event, photographer, UPLOAD, UPLOAD,
        )
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer), event, photographer)
    }

    // A case of the batch with target `action`; APPLIED (current) unless `stage` says otherwise.
    private fun case(b: Batch, action: String, mode: String = "AUTO_HIDE", stage: String = "APPLIED"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, decided_at, applied_mode, action, applied_at) " +
            "VALUES (?, 'QUIET', ?, 1, ?, ?, ?, ?, ?, ?)",
        b.id, UPLOAD, stage, VERDICT.getValue(action), NOW, mode, action, NOW,
    )

    private fun photo(b: Batch, scores: Scores?, status: String = "LIVE", watermark: String? = "reljunk/wm"): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'reljunk/thumb', ?, 125, ?)",
        b.event, b.photographer, "reljunk/${UUID.randomUUID()}", watermark, status,
    ).also { if (scores != null) measure(b, it, scores) }

    private fun measure(b: Batch, photo: UUID, s: Scores) {
        jdbc.update(
            "INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, sharpness, brightness, person_conf, race_conf) " +
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            photo, b.id, NOW, s.fg, s.fg, s.brightness, s.person, s.race,
        )
    }

    private fun hold(b: Batch, photo: UUID, heldBy: String = "AUTO", caseId: UUID? = null) {
        jdbc.update("INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at) VALUES (?, ?, ?, ?, ?)",
            b.id, photo, heldBy, caseId, NOW)
    }

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForList("SELECT held_by, restored_by, count(*) AS n FROM flag_hidden_photos GROUP BY held_by, restored_by ORDER BY 1, 2"),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
        jdbc.queryForObject("SELECT count(*) FROM admin_decision_log", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2008-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
        val GOOD = Scores(fg = 50.0, brightness = 50.0, person = 0.9, race = 0.9)
        val JUNK = Scores(fg = 10.0, brightness = 50.0, person = 0.9, race = 0.9)
        val DARK = Scores(fg = 50.0, brightness = 2.0, person = 0.9, race = 0.9)
        val NOT_RACE = Scores(fg = 50.0, brightness = 50.0, person = 0.1, race = 0.0)
        val VERDICT = mapOf("HIDE_JUNK" to "MIXED", "HIDE_ALL" to "JUNK_DUMP", "NONE" to "FITS")
    }
}
