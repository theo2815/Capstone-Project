package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceHoldService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// RVF-22 (RVR-7): each keyed step of the check, run by two threads at once on a real database; exactly one wins and the
// other changes nothing and throws nothing. The first call's transaction is held open until Postgres shows the second
// waiting on it (pg_blocking_pids), so the two really overlap and the winner is known. The container is shared, so
// fixtures sit in 2010 and assertions look only at this class's own rows.
class RelevanceCheckTwoThreadIntegrationTest : PostgresIntegrationTest() {
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
    fun `two copies claiming one batch - one wins`() {
        val b = batch()
        val (first, second) = (UUID.randomUUID() to UUID.randomUUID())

        val won = race({ batches.claimCheck(b.id, first, NOW.plusMinutes(10), NOW) },
            { tx.execute { batches.claimCheck(b.id, second, NOW.plusMinutes(10), NOW) }!! })

        assertEquals(1 to 0, won)
        assertEquals(first, jdbc.queryForObject("SELECT check_claim FROM batch_relevance WHERE id = ?", UUID::class.java, b.id))
    }

    @Test
    fun `two copies starting one due batch make one case and one sample`() {
        val b = batch()
        repeat(5) { photo(b) }
        val due = due(b)
        val service = service()

        val started = race({ service.start(due, NOW.minusMinutes(10), NOW) }, { service.start(due, NOW.minusMinutes(10), NOW) })

        assertEquals(true to false, started)
        val case = jdbc.queryForObject("SELECT id FROM relevance_cases WHERE batch_id = ?", UUID::class.java, b.id)!!
        assertEquals(5, count("SELECT count(*) FROM relevance_case_photos WHERE case_id = ?", case))
        assertEquals(listOf(null, UPLOAD.toInstant(), 5), row("batch_relevance", b.id, "check_claim", "checked_upload_at", "checked_photos"))
    }

    @Test
    fun `two copies writing one measurement keep one row`() {
        val b = batch()
        val token = claimed(b)
        val p = photo(b)

        val saved = race({ measure(p, b, token, 10.0) }, { tx.execute { measure(p, b, token, 50.0) }!! })

        assertEquals(1 to 1, saved)
        assertEquals(listOf(listOf(50.0, 0.9)), jdbc.queryForList("SELECT sharpness, person_conf FROM relevance_photo_checks WHERE photo_id = ?", p)
            .map { listOf(it["sharpness"], it["person_conf"]) })
    }

    @Test
    fun `two copies finishing one sample store it once`() {
        val b = batch()
        val token = claimed(b)
        val c = case(b, "SAMPLING")

        val finished = race({ finish(c, token, junk = 1) }, { tx.execute { finish(c, token, junk = 2) }!! })

        assertEquals(1 to 0, finished)
        assertEquals(listOf("DECIDING", 1), caseRow(c, "stage", "junk"))
    }

    @Test
    fun `two copies deciding one case store one verdict`() {
        val b = batch()
        val token = claimed(b)
        val c = case(b, "DECIDING")

        val decided = race({ verdict(c, token, "FITS") }, { tx.execute { verdict(c, token, "JUNK_DUMP") }!! })

        assertEquals(1 to 0, decided)
        assertEquals(listOf("DECIDED", "FITS"), caseRow(c, "stage", "verdict"))
    }

    @Test
    fun `two copies applying one decided case make one current case`() {
        val b = batch()
        val c = case(b, "DECIDED", verdict = "JUNK_DUMP")
        val service = service()
        val later = NOW.plusSeconds(1)

        race({ service.applyCase(c, NOW) }, { service.applyCase(c, later) })

        assertEquals(c, cases.currentFor(b.id)?.id)
        // The loser's own time is never written: it changed nothing.
        assertEquals(listOf("APPLIED", "RECORD_ONLY", "HIDE_ALL", NOW.toInstant()), caseRow(c, "stage", "applied_mode", "action", "applied_at"))
        assertEquals(listOf("BIG_HIDE"), jdbc.queryForList("SELECT kind FROM relevance_alerts WHERE case_id = ?", String::class.java, c))
    }

    @Test
    fun `two copies raising one alert store it once`() {
        val b = batch()
        val c = case(b, "DECIDED", verdict = "COPIED")
        val (first, second) = (UUID.randomUUID() to UUID.randomUUID())

        val raised = race({ raise(first, b, c) }, { tx.execute { raise(second, b, c) }!! })

        assertEquals(1 to 0, raised)
        assertEquals(listOf(first), jdbc.queryForList("SELECT id FROM relevance_alerts WHERE case_id = ?", UUID::class.java, c))
    }

    // The first call runs inside a transaction that stays open until the second is seen waiting on it; then it commits.
    private fun <A : Any, B> race(first: () -> A, second: () -> B): Pair<A, B> {
        val pool = Executors.newFixedThreadPool(2)
        val holding = CountDownLatch(1)
        val release = CountDownLatch(1)
        val holderPid = CompletableFuture<Int>()
        try {
            val winner = pool.submit<A> {
                tx.execute {
                    holderPid.complete(pid())
                    first().also {
                        holding.countDown()
                        check(release.await(10, TimeUnit.SECONDS))
                    }
                }!!
            }
            // A first call that threw shows its own error, not a timeout.
            if (!holding.await(10, TimeUnit.SECONDS)) winner.get(1, TimeUnit.SECONDS)
            val loser = pool.submit<B> { second() }
            val holder = holderPid.get(10, TimeUnit.SECONDS)
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
            var blocked = false
            while (!blocked && System.nanoTime() < deadline) {
                blocked = jdbc.queryForObject("SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE ? = ANY(pg_blocking_pids(pid)))",
                    Boolean::class.java, holder)!!
                if (!blocked) Thread.sleep(10)
            }
            assertTrue(blocked, "the second call must wait for the first")
            release.countDown()
            return winner.get(10, TimeUnit.SECONDS) to loser.get(10, TimeUnit.SECONDS)
        } finally {
            release.countDown()
            pool.shutdownNow()
        }
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(): RelevanceCheckService {
        val props = RelevanceProperties(mode = RelevanceMode.RECORD_ONLY, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        @Suppress("UNCHECKED_CAST")
        val provider = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(Mockito.mock(RekognitionAiClient::class.java)) }
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), provider)
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
            holds, photos, decisions, IDLE_POOL, Mockito.mock(RelevanceNotifier::class.java),
            Mockito.mock(RelevanceConfirmer::class.java), Mockito.mock(RelevanceHoldService::class.java))
    }

    // A public shared race's batch, quiet since UPLOAD and never checked: due with the trigger QUIET.
    private fun batch(): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "rel2t-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Two Thread Run', DATE '2010-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "rel2t-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 5, ?, ?)",
            event, photographer, UPLOAD, UPLOAD,
        )
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer), event, photographer)
    }

    private fun due(b: Batch) = object : CheckDue {
        override val batchId = b.id
        override val eventId = b.event
        override val photographerId = b.photographer
        override val lastUploadAt: Instant = UPLOAD.toInstant()
        override val photoCount = 5
        override val trigger = "QUIET"
    }

    private fun claimed(b: Batch): UUID = UUID.randomUUID().also {
        jdbc.update("UPDATE batch_relevance SET check_claim = ?, check_claim_until = ? WHERE id = ?", it, NOW.plusMinutes(10), b.id)
    }

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'rel2t/thumb', 'rel2t/wm', 125, 'LIVE')",
        b.event, b.photographer, "rel2t/${UUID.randomUUID()}",
    )

    private fun case(b: Batch, stage: String, verdict: String? = null): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, band, verdict, decided_at, started_mode) " +
            "VALUES (?, 'QUIET', ?, 600, ?, 300, 30, 'MIXED', ?, ?, 'RECORD_ONLY')",
        b.id, UPLOAD, stage, verdict, verdict?.let { NOW },
    )

    private fun measure(p: UUID, b: Batch, token: UUID, sharp: Double): Int =
        batches.saveMeasurement(p, b.id, token, NOW, sharp, 50.0, sharp, 50.0, 0.9, 0.9)

    private fun finish(c: UUID, token: UUID, junk: Int): Int = cases.finishSampling(c, token, 300, 300, junk, 0, 0, junk, "MIXED")

    private fun verdict(c: UUID, token: UUID, verdict: String): Int =
        cases.saveVerdict(c, token, NOW, verdict, null, verdict, null, null, "summary", "model", "v1", false, false, 0)

    private fun raise(alertId: UUID, b: Batch, c: UUID): Int = alerts.raise(alertId, b.id, c, "COPIED", "copied:$c", true, 600, NOW)

    private fun caseRow(caseId: UUID, vararg columns: String) = row("relevance_cases", caseId, *columns)

    // Timestamps come back as Instants, so they compare with the fixtures' times.
    private fun row(table: String, id: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM $table WHERE id = ?", id)
            .let { r -> columns.map { r[it].let { v -> if (v is java.sql.Timestamp) v.toInstant() else v } } }

    private fun pid(): Int = jdbc.queryForObject("SELECT pg_backend_pid()", Int::class.java)!!

    private fun count(sql: String, vararg params: Any): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2010-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(30)
        val IDLE_POOL = ThreadPoolTaskExecutor().apply { initialize() }
    }
}
