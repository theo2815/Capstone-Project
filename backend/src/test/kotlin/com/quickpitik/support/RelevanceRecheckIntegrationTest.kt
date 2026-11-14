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
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-2.4 (plan part 2, A.3 G7 rows, A.4 G5 flip): the finished full scan may switch Fits and Mixed once, a re-check after
// new uploads changes a hide only as G7 allows, and a batch with an open appeal is not checked again. The container is
// shared, so fixtures sit in 2009 and assertions look only at this class's own rows.
class RelevanceRecheckIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @BeforeEach fun quiet() {
        // Scans other classes left queued would take this class's turns: the container is shared.
        jdbc.update("UPDATE relevance_cases SET scan_state = 'FINISHED' WHERE scan_state IN ('QUEUED', 'RUNNING')")
    }

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch()
        val fits = case(b, "APPLIED", "FITS", applied = "AUTO_HIDE", scan = "QUEUED")
        measured(b, junk = 3, good = 7)
        val dump = batch()
        val old = case(dump, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE")
        val new = case(dump, "DECIDED", "FITS")

        service(mode = RelevanceMode.OFF).tick()

        assertEquals(listOf("FITS", "NONE", "QUEUED", null), row(fits, "verdict", "action", "scan_state", "flipped_at"))
        assertEquals(listOf("APPLIED", "HIDE_ALL"), row(old, "stage", "action"))
        assertEquals(listOf("DECIDED"), row(new, "stage"))
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch()
        case(b, "APPLIED", "FITS", applied = "AUTO_HIDE", scan = "QUEUED")
        measured(b, junk = 3, good = 7)
        val dump = batch()
        val old = case(dump, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE")
        val new = case(dump, "DECIDED", "FITS")
        val before = untouched()

        service(mode = RelevanceMode.RECORD_ONLY).tick()

        assertEquals(before, untouched())
        assertEquals(listOf("CLOSED", "HELD_BY_CASE"), row(new, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "HIDE_ALL"), row(old, "stage", "action"))
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch()
        val fits = case(b, "APPLIED", "FITS", applied = "RECORD_ONLY", scan = "QUEUED")
        measured(b, junk = 3, good = 7)
        val dump = batch()
        val old = case(dump, "APPLIED", "JUNK_DUMP", applied = "RECORD_ONLY")
        val new = case(dump, "DECIDED", "FITS")
        val service = service()

        service.scanTurns()
        service.applyCase(new, NOW)

        assertEquals(listOf("FITS", "NONE", null), row(fits, "verdict", "action", "flipped_at"))
        // A record-only junk dump hid nothing, so a new verdict replaces it as usual: no lift.
        assertEquals(listOf("CLOSED", "SUPERSEDED"), row(old, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "AUTO_HIDE", "NONE"), row(new, "stage", "applied_mode", "action"))
    }

    @Test
    fun `a fits batch whose full scan ends at ten percent or more becomes mixed and its junk is hidden`() {
        val b = batch()
        val c = case(b, "APPLIED", "FITS", applied = "AUTO_HIDE", scan = "QUEUED")
        val (junk, good) = measured(b, junk = 1, good = 9)
        val service = service()

        service.scanTurns()
        service.reconcileHolds()

        assertEquals(listOf("MIXED", "HIDE_JUNK", "SCAN_FLIP", "FINISHED"), row(c, "verdict", "action", "verdict_note", "scan_state"))
        assertNotNull(row(c, "flipped_at").single())
        assertEquals(listOf("HIDDEN"), junk.map(::status))
        assertEquals(listOf("LIVE"), good.map(::status).distinct())
    }

    @Test
    fun `a mixed batch whose full scan ends under ten percent becomes fits and its photos come back`() {
        val b = batch()
        val c = case(b, "APPLIED", "MIXED", applied = "AUTO_HIDE", scan = "QUEUED")
        val (junk, _) = measured(b, junk = 1, good = 19)
        val service = service()
        service.reconcileHolds()
        assertEquals(listOf("HIDDEN"), junk.map(::status))

        service.scanTurns()
        service.reconcileHolds()

        assertEquals(listOf("FITS", "NONE", "SCAN_FLIP"), row(c, "verdict", "action", "verdict_note"))
        assertEquals(listOf("LIVE"), junk.map(::status))
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `a case flips at most once`() {
        val b = batch()
        val c = case(b, "APPLIED", "FITS", applied = "AUTO_HIDE", scan = "QUEUED")
        val (junk, _) = measured(b, junk = 1, good = 9)
        val service = service()
        service.scanTurns()
        val flippedAt = row(c, "flipped_at").single()
        assertEquals("MIXED", row(c, "verdict").single())

        // The scan is queued again and now ends clean: once flipped, the case stays as it is.
        jdbc.update("UPDATE relevance_photo_checks SET fg_sharpness = 50, sharpness = 50 WHERE photo_id = ?", junk.single())
        tx.execute { cases.queueScan(c) }
        service.scanTurns()

        assertEquals(listOf("MIXED", "HIDE_JUNK", "FINISHED", 0), row(c, "verdict", "action", "scan_state", "scan_junk"))
        assertEquals(flippedAt, row(c, "flipped_at").single())
    }

    @Test
    fun `a mixed batch that ends over seventy percent stays mixed`() {
        val b = batch()
        val c = case(b, "APPLIED", "MIXED", applied = "AUTO_HIDE", scan = "QUEUED")
        measured(b, junk = 8, good = 2)

        service().scanTurns()

        assertEquals(listOf("MIXED", "HIDE_JUNK", "FINISHED", 8, null), row(c, "verdict", "action", "scan_state", "scan_junk", "flipped_at"))
    }

    @Test
    fun `a scan finish does not flip a case with an open appeal`() {
        val b = batch()
        val c = case(b, "APPLIED", "FITS", applied = "AUTO_HIDE", scan = "QUEUED")
        measured(b, junk = 3, good = 7)
        appeal(b, c)

        service().scanTurns()

        assertEquals(listOf("FITS", "NONE", "FINISHED", null), row(c, "verdict", "action", "scan_state", "flipped_at"))
    }

    @Test
    fun `a junk-dump hide is lifted by a re-check that says fits, and the junk stays hidden`() {
        for (newVerdict in listOf("FITS", "MIXED")) {
            val b = batch()
            val old = case(b, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE")
            val (junk, good) = measured(b, junk = 1, good = 1)
            val failed = photo(b).also { jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measure_error) VALUES (?, ?, 'GAVE_UP')", it, b.id) }
            val unmeasured = photo(b)
            val service = service()
            service.reconcileHolds()
            assertEquals(listOf("HIDDEN"), (junk + good + failed + unmeasured).map(::status).distinct(), newVerdict)

            val new = case(b, "DECIDED", newVerdict)
            service.applyCase(new, NOW)
            service.reconcileHolds()

            assertEquals(listOf("CLOSED", "RECHECK_LIFTED"), row(old, "stage", "close_reason"), newVerdict)
            assertEquals(listOf("APPLIED", "AUTO_HIDE", "HIDE_JUNK", "QUEUED"), row(new, "stage", "applied_mode", "action", "scan_state"), newVerdict)
            assertEquals(listOf("HIDDEN", "LIVE", "LIVE", "HIDDEN"), (junk + good + failed + unmeasured).map(::status), newVerdict)
        }
    }

    @Test
    fun `a wrong-event hide is not lifted by a re-check`() {
        val held = listOf("WRONG_EVENT" to "FITS", "WRONG_EVENT" to "MIXED", "WRONG_EVENT" to "JUNK_DUMP",
            "COPIED" to "FITS", "COPIED" to "MIXED", "COPIED" to "JUNK_DUMP", "COPIED" to "WRONG_EVENT")
        for ((oldVerdict, newVerdict) in held) {
            for ((label, terms) in listOf("auto-hide" to "1.7", "no consent" to "1.0")) {
                val b = batch(terms = terms)
                val old = case(b, "APPLIED", oldVerdict, applied = "AUTO_HIDE")
                val new = case(b, "DECIDED", newVerdict)

                service().applyCase(new, NOW)

                assertEquals(listOf("CLOSED", "HELD_BY_CASE"), row(new, "stage", "close_reason"), "$oldVerdict -> $newVerdict, $label")
                assertEquals(listOf("APPLIED", "HIDE_ALL"), row(old, "stage", "action"), "$oldVerdict -> $newVerdict, $label")
            }
        }
        // A whole-batch hide that turns into a stronger finding is replaced (A.3).
        for ((oldVerdict, newVerdict) in listOf("WRONG_EVENT" to "COPIED", "JUNK_DUMP" to "WRONG_EVENT", "JUNK_DUMP" to "COPIED")) {
            val b = batch()
            val old = case(b, "APPLIED", oldVerdict, applied = "AUTO_HIDE")
            val new = case(b, "DECIDED", newVerdict)
            service().applyCase(new, NOW)
            assertEquals(listOf("CLOSED", "SUPERSEDED"), row(old, "stage", "close_reason"), "$oldVerdict -> $newVerdict")
            assertEquals(listOf("APPLIED", "HIDE_ALL"), row(new, "stage", "action"), "$oldVerdict -> $newVerdict")
        }
    }

    @Test
    fun `a re-check with the same verdict makes no new case`() {
        val b = batch()
        val old = case(b, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE")
        val new = case(b, "DECIDED", "JUNK_DUMP")

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "SAME_VERDICT"), row(new, "stage", "close_reason"))
        assertEquals(old, cases.currentFor(b.id)?.id)
        assertEquals(listOf("APPLIED", "HIDE_ALL"), row(old, "stage", "action"))
    }

    @Test
    fun `a batch with an open appeal is not checked again`() {
        val open = batch(upload = NOW.minusMinutes(30))
        appeal(open, case(open, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE"))
        val decided = batch(upload = NOW.minusMinutes(30))
        appeal(decided, case(decided, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE"), status = "LOST")

        val due = batches.findCheckDue(NOW.minusMinutes(10), NOW, 500, 100_000).mapNotNull { it.batchId }

        assertFalse(open.id in due)
        assertTrue(decided.id in due, "a decided appeal does not stop the re-check")
    }

    @Test
    fun `a lifted case that was not confirmed does not count`() {
        val b = batch()
        val old = case(b, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE")
        val new = case(b, "DECIDED", "FITS")

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "RECHECK_LIFTED", null), row(old, "stage", "close_reason", "confirmed_at"))
        assertEquals(0L, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE photographer_id = ?", Long::class.java, b.photographer))
    }

    @Test
    fun `a junk-only hide writes its totals once, at the real count, after the scan finishes`() {
        val b = batch()
        val c = case(b, "APPLIED", "MIXED", applied = "AUTO_HIDE", scan = "FINISHED")
        measured(b, junk = 2, good = 3)
        val service = service(chunkSize = 1, chunksPerTick = 1, bigHidePhotos = 2)

        repeat(5) { service.reconcileHolds() }

        assertEquals(2, row(c, "hidden_photos").single())
        assertNotNull(row(c, "hide_finished_at").single())
        assertEquals(listOf(listOf("batch_auto_hidden", true, c.toString(), "2")), decisionsOf(b))
        assertEquals(listOf(listOf("BIG_HIDE", "big_hide:$c", false, 2)), alertsOf(b))
    }

    @Test
    fun `no totals are written before the scan finishes, or for a junk-only hide of no photos`() {
        val running = batch()
        val r = case(running, "APPLIED", "MIXED", applied = "AUTO_HIDE", scan = "RUNNING")
        val (junk, _) = measured(running, junk = 2, good = 3)
        val empty = batch()
        val e = case(empty, "APPLIED", "MIXED", applied = "AUTO_HIDE", scan = "FINISHED")
        measured(empty, junk = 0, good = 3)
        val service = service(bigHidePhotos = 1)

        repeat(2) { service.reconcileHolds() }

        assertEquals(listOf("HIDDEN"), junk.map(::status).distinct())
        for ((b, c) in listOf(running to r, empty to e)) {
            assertEquals(listOf(0, null), row(c, "hidden_photos", "hide_finished_at"))
            assertEquals(emptyList(), decisionsOf(b))
            assertEquals(emptyList(), alertsOf(b))
        }
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(
        mode: RelevanceMode = RelevanceMode.AUTO_HIDE,
        chunkSize: Int = 1000,
        chunksPerTick: Int = 10,
        bigHidePhotos: Int = 500,
    ): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
            alerts = RelevanceProperties.Alerts(bigHidePhotos = bigHidePhotos),
            hide = RelevanceProperties.Hide(chunkSize = chunkSize, chunksPerTick = chunksPerTick),
            scan = RelevanceProperties.Scan(maxBatchesPerTick = 100))
        @Suppress("UNCHECKED_CAST")
        val provider = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(Mockito.mock(RekognitionAiClient::class.java)) }
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), provider)
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
            holds, photos, decisions, IDLE_POOL)
    }

    // A public shared race's batch, checked and scored at `upload`, whose photographer accepted `terms`.
    private fun batch(terms: String = "1.7", upload: OffsetDateTime = UPLOAD): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', ?)",
            "relrecheck-${UUID.randomUUID()}@test.local", terms,
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Recheck Run', DATE '2009-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relrecheck-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 600, ?, ?)",
            event, photographer, upload, upload,
        )
        val batchId = id(
            "INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, copy_low, checked_upload_at, checked_photos) " +
                "VALUES (?, ?, ?, ?, false, ?, 600)",
            event, photographer, NOW, UPLOAD, UPLOAD,
        )
        return Batch(batchId, event, photographer)
    }

    private fun case(b: Batch, stage: String, verdict: String, applied: String? = null, scan: String = "NONE"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, band, verdict, copy_low, " +
            "decided_at, applied_mode, action, applied_at, scan_state) VALUES (?, 'QUIET', ?, 600, ?, 300, 30, 'MIXED', ?, false, ?, ?, ?, ?, ?)",
        b.id, UPLOAD, stage, verdict, NOW, applied, applied?.let { action(verdict) }, applied?.let { NOW }, scan,
    )

    private fun action(verdict: String) = when (verdict) {
        "MIXED" -> "HIDE_JUNK"
        "JUNK_DUMP", "WRONG_EVENT", "COPIED" -> "HIDE_ALL"
        else -> "NONE"
    }

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'relrecheck/thumb', 'relrecheck/wm', 125, 'LIVE')",
        b.event, b.photographer, "relrecheck/${UUID.randomUUID()}",
    )

    // `junk` blurry photos and `good` clean ones, all measured (blur cut-off 20).
    private fun measured(b: Batch, junk: Int, good: Int): Pair<List<UUID>, List<UUID>> {
        fun one(sharp: Double) = photo(b).also {
            jdbc.update(
                "INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, sharpness, brightness, person_conf, race_conf) " +
                    "VALUES (?, ?, ?, ?, ?, 50, 0.9, 0.9)", it, b.id, NOW, sharp, sharp,
            )
        }
        return List(junk) { one(10.0) } to List(good) { one(50.0) }
    }

    private fun appeal(b: Batch, caseId: UUID, status: String = "OPEN") {
        jdbc.update("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason, status) VALUES (?, ?, ?, 'mine', ?)",
            caseId, b.id, b.photographer, status)
    }

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun alertsOf(b: Batch): List<List<Any?>> =
        jdbc.queryForList("SELECT kind, dedupe_key, would_have, photos FROM relevance_alerts WHERE batch_id = ? ORDER BY kind", b.id)
            .map { listOf(it["kind"], it["dedupe_key"], it["would_have"], it["photos"]) }

    private fun decisionsOf(b: Batch): List<List<Any?>> =
        jdbc.queryForList("SELECT decision, automatic, meta->>'caseId' AS case_id, meta->>'photos' AS photos FROM admin_decision_log WHERE target_user_id = ?",
            b.photographer).map { listOf(it["decision"], it["automatic"], it["case_id"], it["photos"]) }

    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForList("SELECT held_by, restored_by, count(*) AS n FROM flag_hidden_photos GROUP BY held_by, restored_by ORDER BY 1, 2"),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
        jdbc.queryForObject("SELECT count(*) FROM admin_decision_log", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2009-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
        val IDLE_POOL = ThreadPoolTaskExecutor().apply { initialize() }
    }
}
