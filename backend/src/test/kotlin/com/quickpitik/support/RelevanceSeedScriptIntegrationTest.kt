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
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
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

    @BeforeEach
    fun copyScripts() {
        for (script in listOf(SEED, CLEANUP)) {
            postgres.copyFileToContainer(MountableFile.forHostPath(Path.of("scripts/$script").toAbsolutePath()), "/tmp/$script")
        }
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
    fun `the seed refuses a batch that already has a case, a score or a word`() {
        val dirt = listOf<(Batch) -> Unit>(
            { b -> ownCase(b) },
            { b -> ownScore(b) },
            { b -> ownWord(b) },
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

    private data class Batch(val event: UUID, val photographer: UUID)

    private fun seed(targets: Map<String, Batch>): Container.ExecResult {
        val args = mutableListOf("psql", "-X", "-U", "quickpitik", "-d", "quickpitik_test")
        for ((verdict, b) in targets) args += listOf("-v", "${verdict.lowercase()}=${b.event}/${b.photographer}")
        args += listOf("-f", "/tmp/$SEED")
        return postgres.execInContainer(*args.toTypedArray())
    }

    private fun cleanup(): Container.ExecResult =
        postgres.execInContainer("psql", "-X", "-U", "quickpitik", "-d", "quickpitik_test", "-f", "/tmp/$CLEANUP")

    private fun sixBatches(): Map<String, Batch> = VERDICTS.associateWith { batch() }

    // A public event with one photographer's photos, as a staging batch looks: no relevance rows of its own yet.
    private fun batch(): Batch {
        val photographer = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Seed', 'PHOTOGRAPHER')",
            "relseed-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Seed Run', DATE '2003-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relseed-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?, ?)",
            event, photographer, PHOTOS, NOW, NOW)
        repeat(PHOTOS) {
            jdbc.update("INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, 'LIVE')",
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
            org.springframework.core.task.SyncTaskExecutor())
    }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private fun count(sql: String, vararg params: Any?): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private companion object {
        const val SEED = "relevance-v2-seed.sql"
        const val CLEANUP = "relevance-v2-seed-cleanup.sql"
        const val PHOTOS = 12
        val VERDICTS = listOf("FITS", "MIXED", "JUNK_DUMP", "WRONG_EVENT", "COPIED", "CANT_TELL")
        val NOW: OffsetDateTime = OffsetDateTime.parse("2003-03-04T05:06:07Z")
    }
}
