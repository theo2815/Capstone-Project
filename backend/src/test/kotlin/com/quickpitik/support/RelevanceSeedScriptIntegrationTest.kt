package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceHoldService
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import org.testcontainers.containers.Container
import org.testcontainers.utility.MountableFile
import java.nio.file.Path
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * RV-1.12 (G15): runs `scripts/relevance-v2-seed.sql` and `scripts/relevance-v2-seed-cleanup.sql` with psql inside
 * the test database container. The container is shared, so fixtures sit in 2003 and assertions look only at this
 * class's own batches.
 */
class RelevanceSeedScriptIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: com.quickpitik.repository.RelevanceHoldRepository
    @Autowired private lateinit var photos: com.quickpitik.repository.PhotoRepository
    @Autowired private lateinit var strikes: com.quickpitik.repository.RelevanceStrikeRepository
    @Autowired private lateinit var users: com.quickpitik.repository.UserRepository
    @Autowired private lateinit var decisions: com.quickpitik.service.admin.AdminDecisionLogService
    @Autowired private lateinit var systemFlags: com.quickpitik.service.admin.SystemFlagService
    @Autowired private lateinit var adminFlags: com.quickpitik.service.admin.AdminFlagService
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)

    @BeforeEach
    fun copyScripts() {
        for (script in listOf(SEED, CLEANUP)) {
            postgres.copyFileToContainer(MountableFile.forHostPath(Path.of("scripts/$script").toAbsolutePath()), "/tmp/$script")
        }
    }

    // Leaves no seeded payout hold behind, even after a failed test: a later clean-up in this shared database would refuse.
    @AfterEach
    fun endSeededHolds() {
        val held = jdbc.queryForList("SELECT c.id FROM relevance_cases c JOIN flags f ON f.id = c.hold_flag_id " +
            "WHERE c.trigger = 'SEEDED' AND f.status IN ('open', 'escalated') AND f.payout_hold_at IS NOT NULL", UUID::class.java)
        if (held.isEmpty()) return
        val admin = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Adm', 'ADMIN')", "relseed-admin-${UUID.randomUUID()}@test.local")
        held.forEach { c -> tx.execute { hold().end(c, admin, restore = false) } }
    }

    @Test
    fun `the seed makes one decided case per verdict and the tick applies them`() {
        val targets = sixBatches()

        val result = seed(targets)

        assertEquals(0, result.exitCode, result.stderr)
        val seeded = seededCases(targets.values)
        assertEquals(VERDICTS.toSet(), seeded.keys)
        for ((verdict, b) in targets) {
            val row = jdbc.queryForMap("SELECT stage, admin_reason, batch_photos, measured FROM relevance_cases WHERE id = ?", seeded.getValue(verdict))
            assertEquals("DECIDED", row["stage"], verdict)
            assertTrue((row["admin_reason"] as String).startsWith("SEEDED:"), verdict)
            assertEquals(PHOTOS, row["batch_photos"], verdict)
            assertEquals(PHOTOS, count("SELECT count(*) FROM relevance_photo_checks WHERE batch_id = ? AND measured_at IS NOT NULL", batchId(b)), verdict)
            assertTrue(count("SELECT count(*) FROM batch_relevance_words WHERE batch_id = ?", batchId(b)) > 0, verdict)
        }
        assertTrue(result.stdout.contains("SEEDED"), result.stdout)

        service().tick()

        for ((verdict, id) in seeded) {
            assertEquals("APPLIED", jdbc.queryForObject("SELECT stage FROM relevance_cases WHERE id = ?", String::class.java, id), verdict)
        }
        assertEquals("HIDE_JUNK", action(seeded.getValue("MIXED")))
        assertEquals("NONE", action(seeded.getValue("FITS")))
        assertEquals(listOf("COPIED"), jdbc.queryForList("SELECT kind FROM relevance_alerts WHERE case_id = ? AND would_have", String::class.java,
            seeded.getValue("COPIED")))
    }

    @Test
    fun `the seed refuses a batch that already has a case, a score, a word or a hold`() {
        val dirt = listOf<(Batch) -> Unit>(
            { b -> ownCase(b) },
            { b -> ownScore(b) },
            { b -> ownWord(b) },
            { b -> ownHold(b) },
        )
        for (makeDirty in dirt) {
            val targets = sixBatches()
            makeDirty(targets.getValue("COPIED"))

            val result = seed(targets)

            assertNotEquals(0, result.exitCode, result.stdout)
            assertTrue(result.stderr.contains("Refused"), result.stderr)
            assertEquals(emptyMap(), seededCases(targets.values), "nothing is seeded when one batch is refused")
            assertEquals(0, count("SELECT count(*) FROM relevance_photo_checks WHERE batch_id = ?", batchId(targets.getValue("FITS"))))
        }
    }

    @Test
    fun `a batch that was never seeded keeps its scores and words`() {
        val outsider = batch()
        ownScore(outsider); ownWord(outsider)
        val before = outsiderRows(outsider)

        assertEquals(0, seed(sixBatches()).exitCode)
        assertEquals(before, outsiderRows(outsider))
        assertEquals(0, cleanup().exitCode)
        assertEquals(before, outsiderRows(outsider))
    }

    @Test
    fun `the clean-up removes every seeded row and nothing else`() {
        val targets = sixBatches()
        assertEquals(0, seed(targets).exitCode)
        service().tick()
        val outsider = batch()
        ownScore(outsider); ownWord(outsider)
        val outsiderCase = ownCase(outsider)
        jdbc.update("INSERT INTO relevance_alerts (batch_id, case_id, kind, dedupe_key, would_have, photos, created_at) " +
            "VALUES (?, ?, 'COPIED', ?, true, 40, ?)", batchId(outsider), outsiderCase, "copied:$outsiderCase", NOW)
        val before = outsiderRows(outsider)

        val result = cleanup()

        assertEquals(0, result.exitCode, result.stderr)
        val ids = targets.values.map { batchId(it) }
        for (table in listOf("relevance_cases", "relevance_alerts", "relevance_photo_checks", "batch_relevance_words")) {
            assertEquals(0, ids.sumOf { count("SELECT count(*) FROM $table WHERE batch_id = ?", it) }, table)
        }
        assertEquals(before, outsiderRows(outsider))
        assertTrue(result.stdout.contains("relevance_cases"), result.stdout)
    }

    // RV-4.4: the auto-hide variant. One photographer has four whole-batch batches (junk dump, wrong event and two
    // copied); a seventh batch belongs to a photographer who withdrew the consent to automatic actions (RV-L7a).
    @Test
    fun `the auto-hide seed leads to hidden batches, notes and alerts after the ticks`() {
        val (targets, seeded) = autoHideRun()

        assertEquals(listOf(MIN_VERSION), targets.values.map { it.photographer }.distinct().map {
            jdbc.queryForObject("SELECT terms_version FROM users WHERE id = ?", String::class.java, it) }.distinct())
        for (slot in listOf("JUNK_DUMP", "WRONG_EVENT", "COPIED", "COPIED_2")) {
            val c = seeded.getValue(slot)
            assertEquals(listOf("AUTO_HIDE", "HIDE_ALL"), caseRow(c, "applied_mode", "action"), slot)
            assertEquals(listOf("HIDDEN"), statuses(targets.getValue(slot)).distinct(), slot)
            assertEquals(1, count("SELECT count(*) FROM admin_decision_log WHERE decision = 'batch_auto_hidden' AND meta->>'caseId' = ?", c.toString()), slot)
            assertEquals(1, notes(c), slot)
        }
        // Mixed: the 40% junk hidden photo by photo; its hide is stamped once the walk-through finishes its scan by SQL.
        assertEquals(mapOf("HIDDEN" to 4, "LIVE" to 8), statuses(targets.getValue("MIXED")).groupingBy { it }.eachCount())
        assertTrue(caseRow(seeded.getValue("MIXED"), "hide_finished_at").single() != null)
        // Fits: nothing hidden; the finished scan's blurry photos bring the review note.
        assertEquals(listOf("LIVE"), statuses(targets.getValue("FITS")).distinct())
        assertEquals(listOf("AUTO_HIDE", "NONE", 2), caseRow(seeded.getValue("FITS"), "applied_mode", "action", "scan_blurry"))
        assertEquals(1, notes(seeded.getValue("FITS")))
        assertEquals(listOf("AUTO_HIDE", "NONE"), caseRow(seeded.getValue("CANT_TELL"), "applied_mode", "action"))
        // The photographer who withdrew: applied as "would have hidden", nothing hidden, told nothing.
        assertEquals(listOf("RECORD_ONLY", "HIDE_ALL", "NO_CONSENT"), caseRow(seeded.getValue("WITHDRAWN"), "applied_mode", "action", "verdict_note"))
        assertEquals(listOf("LIVE"), statuses(targets.getValue("WITHDRAWN")).distinct())
        assertEquals(0, notes(seeded.getValue("WITHDRAWN")))
        // Each copied case holds payouts on a flag of its own and raises its alert.
        for (slot in listOf("COPIED", "COPIED_2")) {
            val c = seeded.getValue(slot)
            assertTrue(caseRow(c, "hold_flag_id").single() != null, slot)
            assertEquals(listOf("COPIED"), jdbc.queryForList("SELECT kind FROM relevance_alerts WHERE case_id = ?", String::class.java, c), slot)
        }
    }

    @Test
    fun `the clean-up puts every photo back`() {
        val (targets, seeded) = autoHideRun()
        // As the walk-through does: back-date the whole-batch notes past the appeal window, so the cases are confirmed.
        jdbc.update("UPDATE relevance_cases SET notice_at = notice_at - interval '16 days' WHERE id IN (?, ?, ?)",
            seeded.getValue("JUNK_DUMP"), seeded.getValue("WRONG_EVENT"), seeded.getValue("COPIED"))
        autoHide().confirmDue()
        assertEquals(3, count("SELECT count(*) FROM relevance_strikes WHERE photographer_id = ?", targets.getValue("COPIED").photographer))
        endHolds(seeded)

        val result = cleanup()

        assertEquals(0, result.exitCode, result.stderr)
        for ((slot, b) in targets) assertEquals(listOf("LIVE"), statuses(b).distinct(), slot)
        val ids = targets.values.map { batchId(it) }
        for (table in listOf("relevance_cases", "flag_hidden_photos", "relevance_alerts")) {
            assertEquals(0, ids.sumOf { count("SELECT count(*) FROM $table WHERE batch_id = ?", it) }, table)
        }
        assertEquals(0, targets.values.sumOf { count("SELECT count(*) FROM relevance_strikes WHERE photographer_id = ?", it.photographer) })
        assertEquals(0, targets.values.sumOf {
            count("SELECT count(*) FROM photographer_messages WHERE photographer_id = ? AND kind = 'relevance_notice'", it.photographer) })
    }

    @Test
    fun `the clean-up refuses while a seeded copied case still holds payouts`() {
        val (targets, _) = autoHideRun()

        val result = cleanup()

        assertNotEquals(0, result.exitCode, result.stdout)
        assertTrue(result.stderr.contains("Refused"), result.stderr)
        assertEquals(listOf("HIDDEN"), statuses(targets.getValue("COPIED")).distinct(), "nothing is cleaned when it refuses")
    }

    private data class Batch(val event: UUID, val photographer: UUID)

    // The auto-hide walk-through on this database: the owner's withdrawal, the seed, the ticks, the scans finished by
    // SQL, and the ticks again. Returns the batches and the seeded case of each slot.
    private fun autoHideRun(): Pair<Map<String, Batch>, Map<String, UUID>> {
        val repeat = photographer()
        val targets = linkedMapOf(
            "FITS" to batch(), "MIXED" to batch(), "CANT_TELL" to batch(), "WITHDRAWN" to batch(),
            "JUNK_DUMP" to batch(repeat), "WRONG_EVENT" to batch(repeat), "COPIED" to batch(repeat), "COPIED_2" to batch(repeat),
        )
        jdbc.update("UPDATE users SET relevance_consent_withdrawn_at = now() WHERE id = ?", targets.getValue("WITHDRAWN").photographer)

        val result = seed(targets, "-v", "terms_version=$MIN_VERSION")
        assertEquals(0, result.exitCode, result.stderr)
        val seeded = targets.mapValues { (_, b) ->
            jdbc.queryForObject("SELECT id FROM relevance_cases WHERE batch_id = ? AND trigger = 'SEEDED'", UUID::class.java, batchId(b))!!
        }
        val service = autoHide()
        val notifier = notifier()
        fun ticks() = repeat(2) {
            seeded.values.forEach { service.applyCase(it, OffsetDateTime.now()) }
            service.reconcileHolds()
            notifier.notifyOwed()
            service.placeOwedHolds()
        }
        ticks()
        for (slot in listOf("FITS", "MIXED")) assertEquals(1, jdbc.update("$FINISH_SCANS AND id = ?", seeded.getValue(slot)), slot)
        ticks()
        return targets to seeded
    }

    private fun endHolds(seeded: Map<String, UUID>) {
        val admin = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Adm', 'ADMIN')", "relseed-admin-${UUID.randomUUID()}@test.local")
        for (slot in listOf("COPIED", "COPIED_2")) assertTrue(tx.execute { hold().end(seeded.getValue(slot), admin, restore = false) }!!, slot)
    }

    private fun autoHideProps() = RelevanceProperties(mode = RelevanceMode.AUTO_HIDE,
        junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0), check = RelevanceProperties.Check(maxBatchesPerTick = 10_000))

    private fun switch(): RelevanceSwitch = RelevanceSwitch(autoHideProps(), AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false))

    private fun notifier() = RelevanceNotifier(cases, decisions, switch(), autoHideProps(), Clock.systemUTC(), tx, publisher)

    private fun hold() = RelevanceHoldService(cases, strikes, systemFlags, adminFlags, decisions, switch(), tx, AdminProperties(flagsEnabled = true), users)

    // Auto-hide with AI off, as the walk-through runs staging: only the database-only steps run.
    private fun autoHide(): RelevanceCheckService {
        val props = autoHideProps()
        return RelevanceCheckService(batches, cases, props, switch(), Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts, publisher,
            LegalProperties(currentVersion = MIN_VERSION, photographerMinVersion = MIN_VERSION), holds, photos, decisions,
            org.springframework.core.task.SyncTaskExecutor(), notifier(),
            RelevanceConfirmer(cases, strikes, alerts, events, notifier(), props, publisher), hold())
    }

    private fun statuses(b: Batch): List<String> =
        jdbc.queryForList("SELECT status FROM photos WHERE event_id = ? AND photographer_id = ?", String::class.java, b.event, b.photographer)

    private fun caseRow(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun notes(caseId: UUID): Int = count("SELECT count(*) FROM photographer_messages WHERE relevance_case_id = ?", caseId)

    private fun seed(targets: Map<String, Batch>, vararg extra: String): Container.ExecResult {
        val args = mutableListOf("psql", "-X", "-U", "quickpitik", "-d", "quickpitik_test", *extra)
        for ((verdict, b) in targets) args += listOf("-v", "${verdict.lowercase()}=${b.event}/${b.photographer}")
        args += listOf("-f", "/tmp/$SEED")
        return postgres.execInContainer(*args.toTypedArray())
    }

    private fun cleanup(): Container.ExecResult =
        postgres.execInContainer("psql", "-X", "-U", "quickpitik", "-d", "quickpitik_test", "-f", "/tmp/$CLEANUP")

    private fun sixBatches(): Map<String, Batch> = VERDICTS.associateWith { batch() }

    private fun photographer(): UUID = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Seed', 'PHOTOGRAPHER')",
        "relseed-${UUID.randomUUID()}@test.local")

    // A public event with one photographer's photos, as a staging batch looks: no relevance rows of its own yet.
    private fun batch(photographer: UUID = photographer()): Batch {
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Seed Run', DATE '2003-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relseed-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?, ?)",
            event, photographer, PHOTOS, NOW, NOW)
        repeat(PHOTOS) {
            jdbc.update("INSERT INTO photos (event_id, photographer_id, s3_key, watermark_s3_key, price_php, status) VALUES (?, ?, ?, 'relseed/wm', 125, 'LIVE')",
                event, photographer, "relseed/${UUID.randomUUID()}")
        }
        return Batch(event, photographer)
    }

    private fun batchId(b: Batch): UUID {
        jdbc.update("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?) ON CONFLICT DO NOTHING", b.event, b.photographer)
        return jdbc.queryForObject("SELECT id FROM batch_relevance WHERE event_id = ? AND photographer_id = ?", UUID::class.java, b.event, b.photographer)!!
    }

    private fun ownCase(b: Batch): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, applied_mode, action, created_at, decided_at, applied_at) " +
            "VALUES (?, 'QUIET', ?, ?, 'APPLIED', 'FITS', 'RECORD_ONLY', 'NONE', ?, ?, ?)", batchId(b), NOW, PHOTOS, NOW, NOW, NOW)

    private fun ownScore(b: Batch) {
        val photo = jdbc.queryForObject("SELECT id FROM photos WHERE event_id = ? AND photographer_id = ? LIMIT 1", UUID::class.java, b.event, b.photographer)
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, brightness, person_conf, race_conf) " +
            "VALUES (?, ?, ?, 70, 50, 0.9, 0.9)", photo, batchId(b), NOW)
    }

    private fun ownHold(b: Batch) {
        val photo = jdbc.queryForObject("SELECT id FROM photos WHERE event_id = ? AND photographer_id = ? LIMIT 1", UUID::class.java, b.event, b.photographer)
        jdbc.update("INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, hidden_at) VALUES (?, ?, 'PHOTOGRAPHER', ?)", batchId(b), photo, NOW)
    }

    private fun ownWord(b: Batch) {
        jdbc.update("INSERT INTO batch_relevance_words (batch_id, word, photos) VALUES (?, 'OWNWORD', 2)", batchId(b))
    }

    // Every relevance row of a batch, as text, so a change to any of them shows.
    private fun outsiderRows(b: Batch): List<Any?> {
        val id = batchId(b)
        return listOf(
            jdbc.queryForList("SELECT photo_id, measured_at, fg_sharpness FROM relevance_photo_checks WHERE batch_id = ? ORDER BY photo_id", id),
            jdbc.queryForList("SELECT word, photos FROM batch_relevance_words WHERE batch_id = ? ORDER BY word", id),
            jdbc.queryForList("SELECT id, trigger, stage FROM relevance_cases WHERE batch_id = ? ORDER BY id", id),
            jdbc.queryForList("SELECT id, kind FROM relevance_alerts WHERE batch_id = ? ORDER BY id", id),
        ).map { it.toString() }
    }

    private fun seededCases(targets: Collection<Batch>): Map<String, UUID> = targets.flatMap { b ->
        jdbc.queryForList("SELECT verdict, id FROM relevance_cases WHERE batch_id = ? AND trigger = 'SEEDED'", batchId(b))
    }.associate { it["verdict"] as String to it["id"] as UUID }

    private fun action(caseId: UUID) = jdbc.queryForObject("SELECT action FROM relevance_cases WHERE id = ?", String::class.java, caseId)

    // Record-only with AI off, as on staging: only the database-only apply step runs.
    private fun service(): RelevanceCheckService {
        val props = RelevanceProperties(mode = RelevanceMode.RECORD_ONLY, check = RelevanceProperties.Check(maxBatchesPerTick = 10_000))
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = false), AiApiProperties(enabled = false))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.5"),
            Mockito.mock(com.quickpitik.repository.RelevanceHoldRepository::class.java),
            Mockito.mock(com.quickpitik.repository.PhotoRepository::class.java),
            Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java),
            org.springframework.core.task.SyncTaskExecutor(),
            Mockito.mock(com.quickpitik.service.relevance.RelevanceNotifier::class.java),
            Mockito.mock(com.quickpitik.service.relevance.RelevanceConfirmer::class.java),
            Mockito.mock(com.quickpitik.service.relevance.RelevanceHoldService::class.java))
    }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private fun count(sql: String, vararg params: Any?): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private companion object {
        const val SEED = "relevance-v2-seed.sql"
        const val CLEANUP = "relevance-v2-seed-cleanup.sql"
        const val PHOTOS = 12
        // The current legal version (application.yml app.legal), as the walk-through passes it.
        const val MIN_VERSION = "1.7"
        // The walk-through's step that finishes the Fits and Mixed scans by hand (AI is off, so no scan runs).
        const val FINISH_SCANS = "UPDATE relevance_cases SET scan_state = 'FINISHED', scan_finished_at = now(), " +
            "scan_measured = measured, scan_junk = junk, scan_blurry = blurry " +
            "WHERE trigger = 'SEEDED' AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND verdict IN ('FITS', 'MIXED') " +
            "AND scan_state IN ('QUEUED', 'RUNNING')"
        val VERDICTS = listOf("FITS", "MIXED", "JUNK_DUMP", "WRONG_EVENT", "COPIED", "CANT_TELL")
        val NOW: OffsetDateTime = OffsetDateTime.parse("2003-03-04T05:06:07Z")
    }
}
