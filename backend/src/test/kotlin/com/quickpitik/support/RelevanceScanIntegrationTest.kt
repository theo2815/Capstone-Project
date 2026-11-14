package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.JUNK_PREDICATE
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.ai.ImageCheck
import com.quickpitik.service.ai.LabelHit
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.PhotoScores
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.relevance.isJunk
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.core.task.SyncTaskExecutor
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

// RV-2.3a (plan part 2, A.4): the full scan of a Fits or Mixed auto-hide case measures every photo of its batch that
// has no measurement yet, a turn at a time, under the batch claim. AWS is stubbed. The container is shared, so
// fixtures sit in 2008 and assertions look only at this class's own rows.
class RelevanceScanIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var named: NamedParameterJdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    private val client = Mockito.mock(RekognitionAiClient::class.java)

    @BeforeEach fun quiet() {
        // Scans other classes left queued would take this class's turns: the container is shared.
        jdbc.update("UPDATE relevance_cases SET scan_state = 'FINISHED' WHERE scan_state IN ('QUEUED', 'RUNNING')")
        Mockito.`when`(client.imageCheck(anyArg())).thenReturn(ImageCheck(50.0, 60.0, 50.0, 60.0, listOf(LabelHit("Person", 0.9))))
    }
    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch(photos = 2)
        val c = case(b)
        val service = service(mode = RelevanceMode.OFF)

        service.tick()
        service.scanTurns()

        assertEquals(0, measured(b))
        assertEquals(listOf("QUEUED"), row(c, "scan_state"))
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch(photos = 2)
        case(b)
        val before = untouched()

        service(mode = RelevanceMode.RECORD_ONLY).scanTurns()

        assertEquals(before, untouched())
        assertEquals(0, measured(b))
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch(photos = 2)
        val c = case(b, mode = "RECORD_ONLY")

        service().scanTurns()

        assertEquals(0, measured(b))
        assertEquals(listOf("QUEUED"), row(c, "scan_state"))
    }

    @Test
    fun `no scan runs in record-only`() {
        val b = batch(photos = 2)
        val c = case(b)

        service(mode = RelevanceMode.RECORD_ONLY).scanTurns()

        assertEquals(0, measured(b))
        assertEquals(listOf("QUEUED", null), row(c, "scan_state", "scan_served_at"))
    }

    @Test
    fun `the SQL junk rule and the Kotlin junk rule agree on every edge case`() {
        val b = batch(photos = 0)
        // fg sharpness, fg brightness, sharpness, brightness, person, race, error. Cut-offs: blur 20, dark 5, labels 0.6.
        val table = mapOf(
            "blur just under" to PhotoScores(19.99, 50.0, 80.0, 50.0, 0.9, 0.0, null),
            "blur at" to PhotoScores(20.0, 50.0, 80.0, 50.0, 0.9, 0.0, null),
            "blur over" to PhotoScores(20.01, 50.0, 80.0, 50.0, 0.9, 0.0, null),
            "no foreground, whole image under" to PhotoScores(null, null, 19.99, 50.0, 0.9, 0.0, null),
            "no foreground, whole image at" to PhotoScores(null, null, 20.0, 50.0, 0.9, 0.0, null),
            "dark just under" to PhotoScores(50.0, 50.0, 50.0, 4.99, 0.9, 0.0, null),
            "dark at" to PhotoScores(50.0, 50.0, 50.0, 5.0, 0.9, 0.0, null),
            "dark over" to PhotoScores(50.0, 50.0, 50.0, 5.01, 0.9, 0.0, null),
            "labels just under" to PhotoScores(50.0, 50.0, 50.0, 50.0, 0.59, 0.59, null),
            "person at" to PhotoScores(50.0, 50.0, 50.0, 50.0, 0.6, 0.0, null),
            "race at" to PhotoScores(50.0, 50.0, 50.0, 50.0, 0.0, 0.6, null),
            "label confidence 0" to PhotoScores(50.0, 50.0, 50.0, 50.0, 0.0, 0.0, null),
            "measure error with junk scores" to PhotoScores(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, "BAD_IMAGE"),
            "clean" to PhotoScores(80.0, 60.0, 80.0, 60.0, 0.9, 0.9, null),
        )
        val ids = table.mapValues { (_, s) -> photo(b).also { measure(b, it, s) } }
        val props = props()

        val sql = named.queryForList(
            "SELECT k.photo_id FROM relevance_photo_checks k WHERE k.batch_id = :batchId AND $JUNK_PREDICATE",
            mapOf("batchId" to b.id, "blurCutoff" to 20.0, "darkCutoff" to 5.0, "labelMin" to 0.6), UUID::class.java,
        ).toSet()

        for ((name, scores) in table) {
            assertEquals(isJunk(scores, props) == true, ids.getValue(name) in sql, name)
        }
        assertEquals(table.values.count { isJunk(it, props) == true }.toLong(), holds.scanCounts(b.id, 20.0, 5.0, 0.6).junk)
    }

    @Test
    fun `scans take turns, least recently served first`() {
        val (x, y, z) = List(3) { batch(photos = 2) }
        val cx = case(x, appliedAt = NOW.minusMinutes(3))
        val cy = case(y, appliedAt = NOW.minusMinutes(2))
        val cz = case(z, appliedAt = NOW.minusMinutes(1))
        val service = service(photosPerTurn = 1, maxBatchesPerTick = 2)

        service.scanTurns()
        assertEquals(listOf(1, 1, 0), listOf(x, y, z).map(::measured))

        service.scanTurns()
        assertEquals(listOf(2, 1, 1), listOf(x, y, z).map(::measured), "the unserved case, then the one served longest ago")
        listOf(cx, cy, cz).forEach { assertEquals(listOf("RUNNING"), row(it, "scan_state")) }
    }

    @Test
    fun `a photo is not measured twice across the sample and the scan`() {
        val b = batch(photos = 0)
        val sampled = photo(b).also { measure(b, it, PhotoScores(50.0, 50.0, 50.0, 50.0, 0.9, 0.0, null)) }
        repeat(2) { photo(b) }
        case(b)
        val service = service()

        repeat(2) { service.scanTurns() }

        Mockito.verify(client, Mockito.times(2)).imageCheck(anyArg())
        assertEquals(SAMPLED_AT.toInstant(), (check(sampled)["measured_at"] as java.sql.Timestamp).toInstant())
        assertEquals(3, measured(b))
    }

    @Test
    fun `a photo that could not be measured does not keep the scan from finishing`() {
        val b = batch(photos = 0)
        measure(b, photo(b), PhotoScores(50.0, 50.0, 50.0, 50.0, 0.9, 0.0, null))
        val broken = photo(b)
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measure_tries, measure_error) VALUES (?, ?, 5, 'GAVE_UP')", broken, b.id)
        // AWS would still fail on it: tried again, it would keep the scan running for ever.
        Mockito.`when`(client.imageCheck(anyArg())).thenThrow(
            com.quickpitik.service.ai.AiApiException(org.springframework.http.HttpStatus.SERVICE_UNAVAILABLE, null, "throttled"))
        val c = case(b)
        val service = service()

        repeat(2) { service.scanTurns() }

        assertEquals(listOf("FINISHED", 1, 0, 0), row(c, "scan_state", "scan_measured", "scan_junk", "scan_blurry"))
    }

    @Test
    fun `a finished scan stores its counts`() {
        val b = batch(photos = 0)
        // Blurry, dark, not a race, clean, clean; and one trashed blurry photo that does not count.
        listOf(
            PhotoScores(10.0, 50.0, 50.0, 50.0, 0.9, 0.0, null),
            PhotoScores(50.0, 50.0, 50.0, 1.0, 0.9, 0.0, null),
            PhotoScores(50.0, 50.0, 50.0, 50.0, 0.1, 0.1, null),
            PhotoScores(50.0, 50.0, 50.0, 50.0, 0.9, 0.0, null),
            PhotoScores(50.0, 50.0, 50.0, 50.0, 0.9, 0.0, null),
        ).forEach { s -> measure(b, photo(b), s) }
        val trashed = photo(b).also { measure(b, it, PhotoScores(10.0, 50.0, 50.0, 50.0, 0.9, 0.0, null)) }
        jdbc.update("UPDATE photos SET deleted_at = ? WHERE id = ?", NOW, trashed)
        val c = case(b)

        service().scanTurns()

        assertEquals(listOf("FINISHED", 5, 3, 1), row(c, "scan_state", "scan_measured", "scan_junk", "scan_blurry"))
        Mockito.verify(client, Mockito.never()).imageCheck(anyArg())
    }

    @Test
    fun `a finished scan queued again measures the new photos and counts each photo once`() {
        val b = batch(photos = 2)
        val c = case(b)
        val service = service()
        repeat(2) { service.scanTurns() }
        assertEquals(listOf("FINISHED", 2), row(c, "scan_state", "scan_measured"))

        // A same-verdict re-check queues the scan again (RV-2.2a) after a new upload.
        photo(b)
        jdbc.update("UPDATE relevance_cases SET scan_state = 'QUEUED' WHERE id = ?", c)
        repeat(2) { service.scanTurns() }

        assertEquals(listOf("FINISHED", 3), row(c, "scan_state", "scan_measured"))
        Mockito.verify(client, Mockito.times(3)).imageCheck(anyArg())
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun props(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, photosPerTurn: Int = 200, maxBatchesPerTick: Int = 2) =
        RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
            scan = RelevanceProperties.Scan(photosPerTurn = photosPerTurn, maxBatchesPerTick = maxBatchesPerTick))

    private fun service(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, photosPerTurn: Int = 200, maxBatchesPerTick: Int = 2): RelevanceCheckService {
        val props = props(mode, photosPerTurn, maxBatchesPerTick)
        @Suppress("UNCHECKED_CAST")
        val provider = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(client) }
        val storage = Mockito.mock(StorageService::class.java)
        Mockito.`when`(storage.getBytes(anyArg())).thenReturn(byteArrayOf(1, 2, 3))
        val registry = SimpleMeterRegistry()
        val measurer = RelevanceMeasurer(batches, storage, provider, SyncTaskExecutor(), props, Clock.systemUTC(), tx, registry)
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), provider)
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, registry, measurer,
            Mockito.mock(AnthropicClient::class.java), events, alerts, Mockito.mock(ApplicationEventPublisher::class.java),
            LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"), holds, photos, decisions, SyncTaskExecutor())
    }

    private fun batch(photos: Int): Batch {
        val photographer = id("INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relscan-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2008-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relscan-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?, ?)",
            event, photographer, photos, NOW, NOW)
        val batchId = id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer)
        return Batch(batchId, event, photographer).also { b -> repeat(photos) { photo(b) } }
    }

    // A current Fits case of the batch with its scan queued, applied in `mode`.
    private fun case(b: Batch, mode: String = "AUTO_HIDE", appliedAt: OffsetDateTime = NOW): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, decided_at, applied_mode, action, applied_at, scan_state) " +
            "VALUES (?, 'QUIET', ?, 1, 'APPLIED', 'FITS', ?, ?, 'NONE', ?, 'QUEUED')",
        b.id, NOW, NOW, mode, appliedAt,
    )

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, 'LIVE')",
        b.event, b.photographer, "relscan/${UUID.randomUUID()}",
    )

    private fun measure(b: Batch, photo: UUID, s: PhotoScores) = jdbc.update(
        "INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, fg_brightness, sharpness, brightness, " +
            "person_conf, race_conf, measure_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        photo, b.id, if (s.measureError == null) SAMPLED_AT else null, s.fgSharpness, s.fgBrightness, s.sharpness, s.brightness,
        s.personConf, s.raceConf, s.measureError,
    )

    private fun measured(b: Batch): Int = jdbc.queryForObject(
        "SELECT count(*) FROM relevance_photo_checks WHERE batch_id = ? AND measured_at IS NOT NULL", Int::class.java, b.id)!!

    private fun check(photo: UUID): Map<String, Any?> = jdbc.queryForMap("SELECT * FROM relevance_photo_checks WHERE photo_id = ?", photo)

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM flag_hidden_photos", Long::class.java),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
        jdbc.queryForObject("SELECT count(*) FROM relevance_photo_checks", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2008-03-04T05:06:07.123456Z")
        val SAMPLED_AT: OffsetDateTime = NOW.minusDays(1)
    }
}
