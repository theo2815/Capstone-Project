package com.quickpitik.support

import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.DueBatch
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// PE-1.2: the batch queries against real Postgres. The container is shared and findDueBatches is global
// with a limit, so every fixture upload sits in 2001 (the ordering test in 1998), cutoffs come from that
// year, and assertions look only at this class's own batch keys (plan PE-1.2 "Test isolation", R2-76).
class BatchRelevanceQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `ensuring a batch twice makes one row`() {
        val b = batchKey()
        assertEquals(listOf(1, 0), List(2) { ensure(b) })
        assertEquals(1, count("SELECT count(*) FROM batch_relevance WHERE event_id = ? AND photographer_id = ?", b.event, b.photographer))
        assertNotNull(batches.findByEventIdAndPhotographerId(b.event, b.photographer))
    }

    // R2-38: a tick that listed the batch before the anonymise script committed must not make the row again.
    @Test
    fun `ensuring a batch of a suspended photographer or a deleted event makes no row`() {
        val suspended = batchKey(suspended = true)
        val deleted = batchKey(eventDeleted = true)
        assertEquals(0, ensure(suspended))
        assertEquals(0, ensure(deleted))
        assertNull(batches.findByEventIdAndPhotographerId(suspended.event, suspended.photographer))
        assertNull(batches.findByEventIdAndPhotographerId(deleted.event, deleted.photographer))
    }

    @Test
    fun `a batch whose newest upload is under ten minutes old is not due`() {
        val young = batchKey(lastUpload = NOW.minusMinutes(10).plusNanos(1_000))
        val exactlyTen = batchKey(lastUpload = NOW.minusMinutes(10))
        assertFalse(isDue(young))
        assertTrue(isDue(exactlyTen), "the quiet cutoff is <=")
    }

    @Test
    fun `a joined photographer with no upload is never due`() {
        assertFalse(isDue(batchKey(lastUpload = null)))
    }

    @Test
    fun `a photo with no photographer makes no batch`() {
        val b = batchKey()
        jdbc.update("INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, NULL, ?, 125)", b.event, "rel/${UUID.randomUUID()}")
        assertEquals(listOf(b.photographer), due().filter { it.eventId == b.event }.map { it.photographerId })
    }

    @Test
    fun `a batch with no row, a later upload, or pending work past the rescore time is due`() {
        val noRow = batchKey()
        val laterUpload = batchKey().also { scored(it, uploadAt = UPLOAD.minusMinutes(5)) }
        val pendingOld = batchKey().also { scored(it, scoredAt = NOW.minusMinutes(60), metadataPending = 1) }
        val processingOld = batchKey().also { scored(it, scoredAt = NOW.minusMinutes(61), photosProcessing = 2) }
        val pendingRecent = batchKey().also { scored(it, scoredAt = NOW.minusMinutes(59), metadataPending = 1) }

        listOf(noRow, laterUpload, pendingOld, processingOld).forEach { assertTrue(isDue(it), it.toString()) }
        assertFalse(isDue(pendingRecent))
    }

    @Test
    fun `a batch scored for its newest upload with nothing pending is not due`() {
        val b = batchKey()
        val listed = due().single { it.eventId == b.event && it.photographerId == b.photographer }
        assertEquals(UPLOAD.toInstant(), listed.lastUploadAt, "microseconds survive the projection")
        assertEquals(EVENT_DATE, listed.eventDate)
        assertTrue(listed.sharedRace)

        // What PE-1.6 writes after scoring: the values exactly as the due query read them, through the entity.
        ensure(b)
        tx.execute {
            val row = batches.findByEventIdAndPhotographerId(b.event, b.photographer)!!
            row.scoredAt = NOW.minusHours(2)
            row.scoredUploadAt = listed.lastUploadAt.atOffset(ZoneOffset.UTC)
            row.scoredEventDate = listed.eventDate
            row.metadataPending = 0
            row.photosProcessing = 0
            batches.save(row)
        }

        assertFalse(isDue(b))
    }

    @Test
    fun `a changed event date makes a scored batch due again`() {
        val b = batchKey().also { scored(it) }
        assertFalse(isDue(b))
        jdbc.update("UPDATE events SET date = date + 1 WHERE id = ?", b.event)
        assertTrue(isDue(b))
    }

    @Test
    fun `a closed event's batch is due once more and never after its final scoring`() {
        val closedRace = batchKey(eventDate = ADMIN_CLOSED).also { scored(it, eventDate = ADMIN_CLOSED) }
        val openRace = batchKey(eventDate = ADMIN_CLOSED.plusDays(1)).also { scored(it, eventDate = ADMIN_CLOSED.plusDays(1)) }
        val closedOwn = batchKey(eventDate = OWNER_CLOSED, ownEvent = true).also { scored(it, eventDate = OWNER_CLOSED) }
        val openOwn = batchKey(eventDate = OWNER_CLOSED.plusDays(1), ownEvent = true).also { scored(it, eventDate = OWNER_CLOSED.plusDays(1)) }

        assertTrue(isDue(closedRace))
        assertTrue(isDue(closedOwn))
        assertFalse(isDue(openRace))
        assertFalse(isDue(openOwn))

        listOf(closedRace, closedOwn).forEach {
            jdbc.update(
                "UPDATE batch_relevance SET final_scored_at = ? WHERE event_id = ? AND photographer_id = ?",
                NOW.minusHours(1), it.event, it.photographer,
            )
            assertFalse(isDue(it))
        }
    }

    @Test
    fun `batches of a deleted event and of a suspended photographer are never due`() {
        assertFalse(isDue(batchKey(eventDeleted = true)))
        assertFalse(isDue(batchKey(suspended = true)))
    }

    @Test
    fun `due batches come least recently swept first, up to the limit`() {
        val upload = OffsetDateTime.parse("1998-01-01T00:00:00.000001Z")
        val neverA = batchKey(lastUpload = upload)
        val neverB = batchKey(lastUpload = upload.plusSeconds(1))
        val sweptOld = batchKey(lastUpload = upload).also { swept(it, upload.plusDays(1)) }
        val sweptNew = batchKey(lastUpload = upload).also { swept(it, upload.plusDays(2)) }
        val cutoff = upload.plusDays(3)

        fun page(limit: Int) = batches.findDueBatches(cutoff, cutoff, LocalDate.of(1990, 1, 1), LocalDate.of(1990, 1, 1), limit)
            .map { Key(it.eventId, it.photographerId) }

        assertEquals(listOf(neverA, neverB, sweptOld), page(3))
        assertEquals(listOf(neverA, neverB, sweptOld, sweptNew), page(10))
    }

    @Test
    fun `touching a batch stamps the swept time and nothing else`() {
        val b = batchKey().also { scored(it) }
        val row = batches.findByEventIdAndPhotographerId(b.event, b.photographer)!!
        val at = NOW.plusSeconds(1)

        assertEquals(1, tx.execute { batches.touchSwept(row.id, at) })

        val after = batches.findByEventIdAndPhotographerId(b.event, b.photographer)!!
        assertEquals(at.toInstant(), after.lastSweptAt!!.toInstant())
        assertEquals(row.scoredAt!!.toInstant(), after.scoredAt!!.toInstant())
        assertEquals(row.scoredUploadAt!!.toInstant(), after.scoredUploadAt!!.toInstant())
        assertEquals(row.scoredEventDate, after.scoredEventDate)
    }

    @Test
    fun `locking a batch row makes a second transaction wait`() {
        val b = batchKey().also { ensure(it) }
        val id = batches.findByEventIdAndPhotographerId(b.event, b.photographer)!!.id
        val pool = Executors.newFixedThreadPool(2)
        val holding = CountDownLatch(1)
        val release = CountDownLatch(1)
        val holderPid = CompletableFuture<Int>()
        val waiterPid = CompletableFuture<Int>()
        try {
            val holder = pool.submit<Boolean> {
                tx.execute {
                    holderPid.complete(pid())
                    val locked = batches.lockById(id) != null
                    holding.countDown()
                    check(release.await(10, TimeUnit.SECONDS))
                    locked
                }!!
            }
            assertTrue(holding.await(10, TimeUnit.SECONDS))
            val waiter = pool.submit<Boolean> {
                tx.execute {
                    waiterPid.complete(pid())
                    batches.lockById(id) != null
                }!!
            }
            val first = holderPid.get(10, TimeUnit.SECONDS)
            val second = waiterPid.get(10, TimeUnit.SECONDS)
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
            var blocked = false
            while (!blocked && System.nanoTime() < deadline) {
                blocked = jdbc.queryForObject("SELECT ? = ANY(pg_blocking_pids(?))", Boolean::class.java, first, second)!!
                if (!blocked) Thread.sleep(10)
            }
            assertTrue(blocked, "the second lock must wait for the first")
            release.countDown()
            assertTrue(holder.get(10, TimeUnit.SECONDS))
            assertTrue(waiter.get(10, TimeUnit.SECONDS))
        } finally {
            release.countDown()
            pool.shutdownNow()
        }
    }

    private data class Key(val event: UUID, val photographer: UUID)

    private fun batchKey(
        lastUpload: OffsetDateTime? = UPLOAD,
        eventDate: LocalDate = EVENT_DATE,
        ownEvent: Boolean = false,
        eventDeleted: Boolean = false,
        suspended: Boolean = false,
    ): Key {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, suspended_at) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', ?)",
            "rel-${UUID.randomUUID()}@test.local", if (suspended) NOW else null,
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, created_by, deleted_at) VALUES (?, 'Rel Run', ?, 'Cebu', 'ACTIVE', ?, ?)",
            "rel-${UUID.randomUUID()}", eventDate, if (ownEvent) photographer else null, if (eventDeleted) NOW else null,
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?)",
            event, photographer, lastUpload, lastUpload,
        )
        return Key(event, photographer)
    }

    private fun ensure(b: Key): Int = tx.execute { batches.ensureBatch(b.event, b.photographer) }!!

    // A row scored at :scoredAt for the upload at :uploadAt with the given event date and unsettled counts.
    private fun scored(
        b: Key,
        uploadAt: OffsetDateTime = UPLOAD,
        scoredAt: OffsetDateTime = NOW.minusHours(2),
        eventDate: LocalDate = EVENT_DATE,
        metadataPending: Int = 0,
        photosProcessing: Int = 0,
    ) = jdbc.update(
        "INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, scored_event_date, " +
            "metadata_pending, photos_processing) VALUES (?, ?, ?, ?, ?, ?, ?)",
        b.event, b.photographer, scoredAt, uploadAt, eventDate, metadataPending, photosProcessing,
    )

    private fun swept(b: Key, at: OffsetDateTime) = jdbc.update(
        "INSERT INTO batch_relevance (event_id, photographer_id, last_swept_at) VALUES (?, ?, ?)", b.event, b.photographer, at,
    )

    private fun due(): List<DueBatch> =
        batches.findDueBatches(NOW.minusMinutes(10), NOW.minusMinutes(60), ADMIN_CLOSED, OWNER_CLOSED, 10_000)

    private fun isDue(b: Key) = due().any { it.eventId == b.event && it.photographerId == b.photographer }

    private fun pid(): Int = jdbc.queryForObject("SELECT pg_backend_pid()", Int::class.java)!!

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private fun count(sql: String, vararg params: Any): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private companion object {
        // Non-zero microseconds: the due rule compares scored_upload_at with last_upload_at for equality.
        val NOW: OffsetDateTime = OffsetDateTime.parse("2001-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
        val TODAY: LocalDate = LocalDate.of(2001, 3, 4)
        val EVENT_DATE: LocalDate = TODAY
        val ADMIN_CLOSED: LocalDate = TODAY.minusDays(4)
        val OWNER_CLOSED: LocalDate = TODAY.minusMonths(1).minusDays(3)
    }
}
