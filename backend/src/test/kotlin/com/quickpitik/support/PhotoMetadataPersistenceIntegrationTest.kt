package com.quickpitik.support

import com.quickpitik.config.PhotoMetadataProperties
import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.image.PhotoMetadataExtractor
import com.quickpitik.service.photographer.PhotoMetadataService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.StoredObject
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.io.ByteArrayInputStream
import java.nio.file.Path
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import kotlin.test.*
import org.testcontainers.utility.MountableFile

class PhotoMetadataPersistenceIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var metadata: PhotoMetadataRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var properties: PhotoMetadataProperties
    @Autowired private lateinit var clock: Clock
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var transactions: TransactionTemplate
    @Autowired private lateinit var service: PhotoMetadataService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired @Qualifier("watermarkProcessing") private lateinit var watermarkPool: ThreadPoolTaskExecutor
    @MockitoBean private lateinit var extractor: PhotoMetadataExtractor
    @MockitoBean private lateinit var storage: StorageService

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun clearPriorMetadataFixtures() {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while ((watermarkPool.activeCount != 0 || watermarkPool.threadPoolExecutor.queue.isNotEmpty()) && System.nanoTime() < deadline) Thread.sleep(20)
        jdbc.update("DELETE FROM photo_metadata")
    }

    private fun fixture(state: PhotoMetadataState = PhotoMetadataState.PENDING): UUID {
        val event = events.save(Event(slug = "metadata-${UUID.randomUUID()}", name = "Run", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        val photo = photos.save(Photo(eventId = event.id, s3Key = "original-${UUID.randomUUID()}", pricePhp = BigDecimal.ONE, status = PhotoStatus.LIVE))
        metadata.save(PhotoMetadata(photoId = photo.id, originalFilename = "camera.jpg", originalSizeBytes = 42, state = state))
        return photo.id
    }

    private fun claim(id: UUID, token: UUID, now: OffsetDateTime = OffsetDateTime.now()) =
        transactions.execute { metadata.claimIfDue(id, token, 1, now, now.plusMinutes(5), 4) }!!

    private fun old(id: UUID) {
        jdbc.update("UPDATE photos SET uploaded_at = now() - interval '6 minutes' WHERE id = ?", id)
    }

    private fun awaitState(id: UUID, state: PhotoMetadataState, attempts: Int) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (System.nanoTime() < deadline) {
            val row = metadata.findById(id).orElse(null)
            if (row?.state == state && row.attempts == attempts) return
            Thread.sleep(20)
        }
        val row = metadata.findById(id).orElse(null)
        fail("Expected $state/$attempts, got ${row?.state}/${row?.attempts}")
    }

    private fun deferredExecutor(tasks: MutableList<Runnable>, capacity: Int = 1): ThreadPoolTaskExecutor {
        val executor = Mockito.mock(ThreadPoolTaskExecutor::class.java)
        val pool = Mockito.mock(ThreadPoolExecutor::class.java)
        Mockito.`when`(executor.threadPoolExecutor).thenReturn(pool)
        Mockito.`when`(pool.queue).thenReturn(LinkedBlockingQueue<Runnable>())
        Mockito.`when`(executor.maxPoolSize).thenReturn(capacity)
        Mockito.doAnswer { tasks += it.getArgument<Runnable>(0); null }.`when`(executor).execute(Mockito.any(Runnable::class.java))
        return executor
    }

    private fun sweeper(photoRepository: PhotoRepository, executor: ThreadPoolTaskExecutor) =
        PhotoMetadataService(metadata, photoRepository, coverage, extractor, storage, executor, properties, transactions, clock)

    private fun settle(id: UUID, token: UUID, version: Int = 1, model: String = "R6") =
        transactions.execute { metadata.settle(id, token, version, PhotoMetadataState.READY, PhotoMetadataFields(cameraModel = model), OffsetDateTime.now()) }!!

    @Test
    fun `operator seed is bounded scoped and repeat safe without changing photo visibility or counters`() {
        val owner = users.save(User(email = "metadata-seed-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Seed", role = Role.PHOTOGRAPHER))
        val event = events.saveAndFlush(Event(slug = "metadata-seed-${UUID.randomUUID()}", name = "Run", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE, photoCount = 5))
        val otherEvent = events.saveAndFlush(Event(slug = "metadata-seed-${UUID.randomUUID()}", name = "Other", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        coverage.saveAndFlush(EventPhotographer(id = EventPhotographerId(event.id, owner.id), photoCount = 5, firstUploadAt = OffsetDateTime.now(), metadataPublic = true))
        val first = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "seed/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE, status = PhotoStatus.PROCESSING))
        val second = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "seed/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE, status = PhotoStatus.HIDDEN))
        val terminal = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "seed/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE))
        val trash = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "seed/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE))
        val purging = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "seed/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE))
        val foreign = photos.saveAndFlush(Photo(eventId = otherEvent.id, s3Key = "seed/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE))
        jdbc.update("UPDATE photos SET uploaded_at = now() - interval '2 minutes' WHERE id = ?", first.id)
        jdbc.update("UPDATE photos SET uploaded_at = now() - interval '1 minute' WHERE id = ?", second.id)
        jdbc.update("UPDATE photos SET deleted_at = now() WHERE id = ?", trash.id)
        jdbc.update("UPDATE photos SET deleted_at = now(), deletion_started_at = now() WHERE id = ?", purging.id)
        metadata.saveAndFlush(PhotoMetadata(photoId = terminal.id, state = PhotoMetadataState.READY, originalFilename = "preserve.jpg"))

        val script = Path.of("scripts/backfill-photo-metadata.sql").toAbsolutePath()
        PostgresIntegrationTest.postgres.copyFileToContainer(MountableFile.forHostPath(script), "/tmp/backfill-photo-metadata.sql")
        fun runSeed(limit: Int? = null, apply: Boolean? = null, eventId: UUID? = event.id): org.testcontainers.containers.Container.ExecResult {
            val args = mutableListOf("psql", "-X", "-q", "-t", "-A", "-F", "|", "-v", "ON_ERROR_STOP=1", "-U", "quickpitik", "-d", "quickpitik_test")
            if (eventId != null) args += listOf("-v", "event_id=$eventId")
            if (limit != null) args += listOf("-v", "limit=$limit")
            if (apply != null) args += listOf("-v", "apply=$apply")
            args += listOf("-f", "/tmp/backfill-photo-metadata.sql")
            return PostgresIntegrationTest.postgres.execInContainer(*args.toTypedArray())
        }

        assertNotEquals(0, runSeed(2, true, eventId = null).exitCode)
        assertNotEquals(0, runSeed(apply = true).exitCode)
        val dry = runSeed(2)
        assertEquals(0, dry.exitCode, dry.stderr)
        assertEquals(listOf("${first.id}|2", "${second.id}|2"), dry.stdout.trim().lines())
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM photo_metadata WHERE photo_id IN (?, ?)", Int::class.java, first.id, second.id))

        repeat(2) {
            val applied = runSeed(3, true)
            assertEquals(0, applied.exitCode, applied.stderr)
        }
        assertEquals(setOf(first.id, second.id, trash.id, terminal.id), jdbc.queryForList("SELECT photo_id FROM photo_metadata WHERE photo_id IN (?, ?, ?, ?, ?, ?)", UUID::class.java, first.id, second.id, trash.id, terminal.id, purging.id, foreign.id).toSet())
        assertEquals(3, jdbc.queryForObject("SELECT count(*) FROM photo_metadata WHERE photo_id IN (?, ?, ?) AND state = 'PENDING' AND extractor_version = 1 AND attempts = 0 AND original_filename IS NULL", Int::class.java, first.id, second.id, trash.id))
        assertEquals("READY|preserve.jpg", jdbc.queryForObject("SELECT state || '|' || original_filename FROM photo_metadata WHERE photo_id = ?", String::class.java, terminal.id))
        assertEquals("PROCESSING", jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, first.id))
        assertEquals("HIDDEN", jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, second.id))
        assertEquals(5, jdbc.queryForObject("SELECT photo_count FROM events WHERE id = ?", Int::class.java, event.id))
        assertEquals(5, jdbc.queryForObject("SELECT photo_count FROM event_photographer WHERE event_id = ? AND photographer_id = ?", Int::class.java, event.id, owner.id))
        assertEquals(true, jdbc.queryForObject("SELECT metadata_public FROM event_photographer WHERE event_id = ? AND photographer_id = ?", Boolean::class.java, event.id, owner.id))
    }

    @Test
    fun `simultaneous claims yield one worker and one attempt`() {
        val id = fixture()
        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val workers = (1..2).map { pool.submit<Int> { start.await(); claim(id, UUID.randomUUID()) } }
            start.countDown()
            assertEquals(listOf(0, 1), workers.map { it.get(10, TimeUnit.SECONDS) }.sorted())
            val row = metadata.findById(id).orElseThrow()
            assertEquals(PhotoMetadataState.PROCESSING, row.state)
            assertEquals(1, row.attempts)
            assertNotNull(row.leaseToken)
            assertNotNull(row.leaseUntil)
        } finally { pool.shutdownNow() }
    }

    @Test
    fun `expired worker cannot replace newer lease or settle another version`() {
        val id = fixture()
        val old = UUID.randomUUID()
        val newer = UUID.randomUUID()
        val now = OffsetDateTime.now()
        assertEquals(1, claim(id, old, now.minusMinutes(6)))
        assertEquals(1, claim(id, newer, now))
        assertEquals(0, settle(id, old))
        assertEquals(0, settle(id, newer, version = 2))
        assertEquals(1, settle(id, newer, model = "Winning camera"))
        val row = metadata.findById(id).orElseThrow()
        assertEquals("Winning camera", row.fields?.cameraModel)
        assertEquals(2, row.attempts)
        assertNull(row.leaseToken)
        assertNull(row.leaseUntil)
    }

    @Test
    fun `purging prevents claims and settlements and physical purge cannot resurrect metadata`() {
        val id = fixture()
        val token = UUID.randomUUID()
        assertEquals(1, claim(id, token))
        jdbc.update("UPDATE photos SET deleted_at = now(), deletion_started_at = now() WHERE id = ?", id)
        assertEquals(0, settle(id, token))
        jdbc.update("DELETE FROM photos WHERE id = ?", id)
        assertEquals(0, settle(id, token))
        assertEquals(0, claim(id, UUID.randomUUID()))
        assertFalse(metadata.existsById(id))
        val purging = fixture()
        jdbc.update("UPDATE photos SET deleted_at = now(), deletion_started_at = now() WHERE id = ?", purging)
        assertEquals(0, claim(purging, UUID.randomUUID()))
    }

    @Test
    fun `overlapping purge and settlement leave no metadata whichever transaction locks first`() {
        for (purgeFirst in listOf(true, false)) {
            val id = fixture()
            val token = UUID.randomUUID()
            assertEquals(1, claim(id, token))
            val purge = { jdbc.update("DELETE FROM photos WHERE id = ?", id) }
            val settle = {
                metadata.settle(id, token, 1, PhotoMetadataState.READY, PhotoMetadataFields(cameraModel = "R6"), OffsetDateTime.now())
            }
            val holderPid = CompletableFuture<Int>()
            val waiterPid = CompletableFuture<Int>()
            val holdingLock = CountDownLatch(1)
            val releaseLock = CountDownLatch(1)
            val pool = Executors.newFixedThreadPool(2)
            try {
                val holder = pool.submit<Int> {
                    transactions.execute {
                        holderPid.complete(jdbc.queryForObject("SELECT pg_backend_pid()", Int::class.java)!!)
                        val changed = if (purgeFirst) purge() else settle()
                        holdingLock.countDown()
                        check(releaseLock.await(10, TimeUnit.SECONDS))
                        changed
                    }!!
                }
                assertTrue(holdingLock.await(10, TimeUnit.SECONDS))
                val waiter = pool.submit<Int> {
                    transactions.execute {
                        waiterPid.complete(jdbc.queryForObject("SELECT pg_backend_pid()", Int::class.java)!!)
                        if (purgeFirst) settle() else purge()
                    }!!
                }
                val firstPid = holderPid.get(10, TimeUnit.SECONDS)
                val secondPid = waiterPid.get(10, TimeUnit.SECONDS)
                val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
                var blocked = false
                while (!blocked && System.nanoTime() < deadline) {
                    blocked = jdbc.queryForObject(
                        "SELECT ? = ANY(pg_blocking_pids(?))", Boolean::class.java, firstPid, secondPid,
                    )!!
                    if (!blocked) Thread.sleep(10)
                }
                assertTrue(blocked, "Purge and settlement must overlap on the metadata row lock")
                releaseLock.countDown()
                assertEquals(1, holder.get(10, TimeUnit.SECONDS))
                // A committed purge makes the waiting UPDATE affect zero rows;
                // a settlement committed first is removed by the waiting cascade.
                assertEquals(if (purgeFirst) 0 else 1, waiter.get(10, TimeUnit.SECONDS))
                assertFalse(photos.existsById(id))
                assertFalse(metadata.existsById(id))
            } finally {
                releaseLock.countDown()
                pool.shutdownNow()
            }
        }
    }

    @Test
    fun `terminal current version rows skip parsing and future retries and exhausted leases stay untouched`() {
        for (state in listOf(PhotoMetadataState.READY, PhotoMetadataState.EMPTY, PhotoMetadataState.FAILED, PhotoMetadataState.UNAVAILABLE)) {
            val id = fixture(state)
            service.extractIfDue(id, byteArrayOf(1))
            assertEquals(state, metadata.findById(id).orElseThrow().state)
            assertEquals(0, metadata.findById(id).orElseThrow().attempts)
        }
        val future = fixture()
        jdbc.update("UPDATE photo_metadata SET next_attempt_at = now() + interval '1 hour' WHERE photo_id = ?", future)
        assertEquals(0, claim(future, UUID.randomUUID()))
        val exhausted = fixture(PhotoMetadataState.PROCESSING)
        jdbc.update("UPDATE photo_metadata SET attempts = 4, lease_until = now() - interval '1 minute' WHERE photo_id = ?", exhausted)
        assertEquals(0, claim(exhausted, UUID.randomUUID()))
        Mockito.verifyNoInteractions(extractor)
    }

    @Test
    fun `fourth claim can recover an expired metadata lease`() {
        val id = fixture(PhotoMetadataState.PROCESSING)
        jdbc.update("UPDATE photo_metadata SET attempts = 3, lease_until = now() - interval '1 minute' WHERE photo_id = ?", id)
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields(cameraModel = "Recovered"))

        service.extractIfDue(id, byteArrayOf(1))

        val row = metadata.findById(id).orElseThrow()
        assertEquals(PhotoMetadataState.READY, row.state)
        assertEquals(4, row.attempts)
        assertEquals("Recovered", row.fields?.cameraModel)
    }

    @Test
    fun `due selection honors grace leases retries and terminal states`() {
        val oldPending = fixture().also(::old)
        val freshPending = fixture()
        val expired = fixture(PhotoMetadataState.PROCESSING)
        jdbc.update("UPDATE photo_metadata SET lease_until = now() - interval '1 second' WHERE photo_id = ?", expired)
        val active = fixture(PhotoMetadataState.PROCESSING)
        jdbc.update("UPDATE photo_metadata SET lease_until = now() + interval '1 minute' WHERE photo_id = ?", active)
        val dueFailed = fixture(PhotoMetadataState.FAILED)
        jdbc.update("UPDATE photo_metadata SET next_attempt_at = now() - interval '1 second' WHERE photo_id = ?", dueFailed)
        val terminalFailed = fixture(PhotoMetadataState.FAILED)
        val ready = fixture(PhotoMetadataState.READY)
        val empty = fixture(PhotoMetadataState.EMPTY)
        val exhausted = fixture(PhotoMetadataState.FAILED)
        jdbc.update("UPDATE photo_metadata SET attempts = 4, next_attempt_at = now() - interval '1 second' WHERE photo_id = ?", exhausted)

        val now = OffsetDateTime.now()
        val due = metadata.findDueCandidateIds(1, 4, now.minusMinutes(5), now, org.springframework.data.domain.PageRequest.of(0, 10)).toSet()
        assertEquals(setOf(oldPending, expired, dueFailed), due)
        assertFalse(freshPending in due || active in due || terminalFailed in due || ready in due || empty in due || exhausted in due)
    }

    @Test
    fun `recovery reads retained original while watermark photo is still processing`() {
        val id = fixture().also(::old)
        jdbc.update("UPDATE photos SET status = 'PROCESSING' WHERE id = ?", id)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 1, "image/jpeg"))
        Mockito.`when`(storage.open(key)).thenReturn(ByteArrayInputStream(byteArrayOf(1)))
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields(cameraModel = "Recovered"))

        service.recoverDue()
        awaitState(id, PhotoMetadataState.READY, 1)

        assertEquals("Recovered", metadata.findById(id).orElseThrow().fields?.cameraModel)
        assertEquals(PhotoStatus.PROCESSING, photos.findById(id).orElseThrow().status)
        Mockito.verify(storage, Mockito.times(1)).open(key)
    }

    @Test
    fun `recovery fills script seeded file sizes without inventing filename or overwriting known size`() {
        val ready = fixture()
        val empty = fixture()
        for (id in listOf(ready, empty)) {
            jdbc.update("DELETE FROM photo_metadata WHERE photo_id = ?", id)
            jdbc.update("INSERT INTO photo_metadata (photo_id, state, extractor_version) VALUES (?, 'PENDING', 1)", id)
            old(id)
        }
        val known = fixture().also(::old)
        val bytes = mapOf(ready to byteArrayOf(1), empty to byteArrayOf(2), known to byteArrayOf(3))
        for ((id, content) in bytes) {
            val key = photos.findById(id).orElseThrow().s3Key
            Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 1, "image/jpeg"))
            Mockito.`when`(storage.open(key)).thenReturn(ByteArrayInputStream(content))
        }
        Mockito.`when`(extractor.extract(anyArg())).thenAnswer {
            when (it.getArgument<java.io.InputStream>(0).read()) {
                2 -> PhotoMetadataFields()
                else -> PhotoMetadataFields(cameraModel = "R6")
            }
        }
        val tasks = mutableListOf<Runnable>()
        val manual = sweeper(photos, deferredExecutor(tasks))
        for (id in listOf(ready, empty, known)) {
            manual.recoverDue()
            assertEquals(1, tasks.size)
            tasks.removeAt(0).run()
        }

        assertEquals(PhotoMetadataState.READY, metadata.findById(ready).orElseThrow().state)
        assertEquals(PhotoMetadataState.EMPTY, metadata.findById(empty).orElseThrow().state)
        assertEquals(1L, metadata.findById(ready).orElseThrow().originalSizeBytes)
        assertEquals(1L, metadata.findById(empty).orElseThrow().originalSizeBytes)
        assertNull(metadata.findById(ready).orElseThrow().originalFilename)
        assertNull(metadata.findById(empty).orElseThrow().originalFilename)
        assertEquals(42L, metadata.findById(known).orElseThrow().originalSizeBytes)
    }

    @Test
    fun `expired fourth claim becomes terminal without GET while active and renewed leases survive`() {
        val expired = fixture(PhotoMetadataState.PROCESSING)
        val active = fixture(PhotoMetadataState.PROCESSING)
        val renewed = fixture(PhotoMetadataState.PROCESSING)
        val purging = fixture(PhotoMetadataState.PROCESSING)
        val oldToken = UUID.randomUUID()
        val newToken = UUID.randomUUID()
        for (id in listOf(expired, renewed, purging)) {
            jdbc.update("UPDATE photo_metadata SET attempts = 4, lease_token = ?, lease_until = now() - interval '1 minute' WHERE photo_id = ?", oldToken, id)
        }
        jdbc.update("UPDATE photos SET deleted_at = now(), deletion_started_at = now() WHERE id = ?", purging)
        jdbc.update("UPDATE photo_metadata SET attempts = 4, lease_token = ?, lease_until = now() + interval '1 minute' WHERE photo_id = ?", newToken, active)
        old(expired)
        old(active)
        old(renewed)
        val tasks = mutableListOf<Runnable>()
        val manual = sweeper(photos, deferredExecutor(tasks, capacity = 2))

        manual.recoverDue()
        assertEquals(2, tasks.size)
        jdbc.update("UPDATE photo_metadata SET lease_token = ?, lease_until = now() + interval '1 minute' WHERE photo_id = ?", newToken, renewed)
        tasks.forEach(Runnable::run)

        val failed = metadata.findById(expired).orElseThrow()
        assertEquals(PhotoMetadataState.FAILED, failed.state)
        assertEquals(4, failed.attempts)
        assertNull(failed.leaseToken)
        assertNull(failed.leaseUntil)
        assertNull(failed.nextAttemptAt)
        assertEquals(0, settle(expired, oldToken))
        for (id in listOf(active, renewed)) {
            val row = metadata.findById(id).orElseThrow()
            assertEquals(PhotoMetadataState.PROCESSING, row.state)
            assertEquals(4, row.attempts)
            assertEquals(newToken, row.leaseToken)
        }
        assertEquals(PhotoMetadataState.PROCESSING, metadata.findById(purging).orElseThrow().state)
        Mockito.verifyNoInteractions(storage)
    }

    @Test
    fun `transient storage failures schedule three delays then stop after fourth claim`() {
        val id = fixture().also(::old)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenThrow(IllegalStateException("storage down"))
        for ((attempt, delay) in listOf(1L, 5L, 30L, 0L).withIndex()) {
            val before = OffsetDateTime.now()
            service.recoverDue()
            awaitState(id, PhotoMetadataState.FAILED, attempt + 1)
            val row = metadata.findById(id).orElseThrow()
            if (delay == 0L) assertNull(row.nextAttemptAt)
            else {
                assertNotNull(row.nextAttemptAt)
                assertTrue(row.nextAttemptAt!!.isAfter(before.plusMinutes(delay).minusSeconds(5)))
                assertTrue(row.nextAttemptAt!!.isBefore(OffsetDateTime.now().plusMinutes(delay).plusSeconds(5)))
                jdbc.update("UPDATE photo_metadata SET next_attempt_at = now() - interval '1 second' WHERE photo_id = ?", id)
            }
        }
        service.recoverDue()
        assertEquals(4, metadata.findById(id).orElseThrow().attempts)
        Mockito.verify(storage, Mockito.times(4)).metadata(key)
    }

    @Test
    fun `missing and oversized originals settle unavailable before GET`() {
        val missing = fixture().also(::old)
        val oversized = fixture().also(::old)
        val bigKey = photos.findById(oversized).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(bigKey)).thenReturn(StoredObject(bigKey, 25L * 1024 * 1024 + 1, "image/jpeg"))

        service.recoverDue()
        awaitState(missing, PhotoMetadataState.UNAVAILABLE, 1)
        awaitState(oversized, PhotoMetadataState.UNAVAILABLE, 1)
        assertEquals(25L * 1024 * 1024 + 1, metadata.findById(oversized).orElseThrow().originalSizeBytes)
        Mockito.verify(storage, Mockito.never()).open(Mockito.anyString())
    }

    @Test
    fun `deletion between HEAD and GET cannot resurrect metadata`() {
        val id = fixture().also(::old)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 1, "image/jpeg"))
        Mockito.`when`(storage.open(key)).thenAnswer {
            jdbc.update("DELETE FROM photos WHERE id = ?", id)
            ByteArrayInputStream(byteArrayOf(1))
        }

        service.recoverDue()
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (metadata.existsById(id) && System.nanoTime() < deadline) Thread.sleep(20)
        assertFalse(metadata.existsById(id))
    }

    @Test
    fun `object disappearing between HEAD and GET settles unavailable`() {
        val id = fixture().also(::old)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 1, "image/jpeg"), null)
        Mockito.`when`(storage.open(key)).thenThrow(IllegalStateException("object gone"))

        service.recoverDue()
        awaitState(id, PhotoMetadataState.UNAVAILABLE, 1)
        assertNull(metadata.findById(id).orElseThrow().nextAttemptAt)
    }

    @Test
    fun `GET failure and failed recheck remain retryable`() {
        val id = fixture().also(::old)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 1, "image/jpeg"))
            .thenThrow(IllegalStateException("HEAD outage"))
        Mockito.`when`(storage.open(key)).thenThrow(IllegalStateException("GET outage"))

        service.recoverDue()
        awaitState(id, PhotoMetadataState.FAILED, 1)
        assertNotNull(metadata.findById(id).orElseThrow().nextAttemptAt)
    }

    @Test
    fun `short GET settles retryable failure instead of parsing truncated bytes`() {
        val id = fixture().also(::old)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 2, "image/jpeg"))
        Mockito.`when`(storage.open(key)).thenReturn(ByteArrayInputStream(byteArrayOf(1)))
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields())

        service.recoverDue()
        awaitState(id, PhotoMetadataState.FAILED, 1)
        assertNotNull(metadata.findById(id).orElseThrow().nextAttemptAt)
        Mockito.verifyNoInteractions(extractor)
    }

    @Test
    fun `photo lookup failure takes retry schedule and fourth failure stays terminal`() {
        val id = fixture().also(::old)
        val brokenPhotos = Mockito.mock(PhotoRepository::class.java)
        Mockito.`when`(brokenPhotos.findById(id)).thenThrow(IllegalStateException("lookup down"))
        val tasks = mutableListOf<Runnable>()
        val manual = sweeper(brokenPhotos, deferredExecutor(tasks))

        for ((attempt, delay) in listOf(1L, 5L, 30L, 0L).withIndex()) {
            val before = OffsetDateTime.now()
            manual.recoverDue()
            assertEquals(1, tasks.size)
            tasks.removeAt(0).run()
            val row = metadata.findById(id).orElseThrow()
            assertEquals(PhotoMetadataState.FAILED, row.state)
            assertEquals(attempt + 1, row.attempts)
            if (delay == 0L) assertNull(row.nextAttemptAt)
            else {
                assertNotNull(row.nextAttemptAt)
                assertTrue(row.nextAttemptAt!!.isAfter(before.plusMinutes(delay).minusSeconds(5)))
                assertTrue(row.nextAttemptAt!!.isBefore(OffsetDateTime.now().plusMinutes(delay).plusSeconds(5)))
                jdbc.update("UPDATE photo_metadata SET next_attempt_at = now() - interval '1 second' WHERE photo_id = ?", id)
            }
        }
        manual.recoverDue()
        assertTrue(tasks.isEmpty())
        Mockito.verifyNoInteractions(storage)
    }

    @Test
    fun `two sweepers selecting one photo still make one claim and fetch`() {
        val id = fixture().also(::old)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 1, "image/jpeg"))
        Mockito.`when`(storage.open(key)).thenReturn(ByteArrayInputStream(byteArrayOf(1)))
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields(cameraModel = "Winner"))
        val first = mutableListOf<Runnable>()
        val second = mutableListOf<Runnable>()
        sweeper(photos, deferredExecutor(first)).recoverDue()
        sweeper(photos, deferredExecutor(second)).recoverDue()
        assertEquals(1, first.size)
        assertEquals(1, second.size)

        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val workers = listOf(first.single(), second.single()).map { task -> pool.submit { start.await(); task.run() } }
            start.countDown()
            workers.forEach { it.get(10, TimeUnit.SECONDS) }
        } finally { pool.shutdownNow() }
        val row = metadata.findById(id).orElseThrow()
        assertEquals(PhotoMetadataState.READY, row.state)
        assertEquals(1, row.attempts)
        Mockito.verify(storage, Mockito.times(1)).open(key)
    }

    @Test
    fun `deterministic parser failure is terminal in recovery`() {
        val id = fixture().also(::old)
        val key = photos.findById(id).orElseThrow().s3Key
        Mockito.`when`(storage.metadata(key)).thenReturn(StoredObject(key, 1, "image/jpeg"))
        Mockito.`when`(storage.open(key)).thenReturn(ByteArrayInputStream(byteArrayOf(1)))
        Mockito.`when`(extractor.extract(anyArg())).thenThrow(IllegalArgumentException("bad tags"))

        service.recoverDue()
        awaitState(id, PhotoMetadataState.FAILED, 1)
        assertNull(metadata.findById(id).orElseThrow().nextAttemptAt)
        service.recoverDue()
        assertEquals(1, metadata.findById(id).orElseThrow().attempts)
    }

    @Test
    fun `one sweep submits at most idle pool capacity and ten candidates`() {
        val ids = List(12) { fixture().also(::old) }
        val now = OffsetDateTime.now()
        assertEquals(10, metadata.findDueCandidateIds(1, 4, now.minusMinutes(5), now,
            org.springframework.data.domain.PageRequest.of(0, 10)).size)

        service.recoverDue()
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (metadata.findAllById(ids).count { it.state == PhotoMetadataState.UNAVAILABLE } < 2 && System.nanoTime() < deadline) Thread.sleep(20)
        val rows = metadata.findAllById(ids)
        assertEquals(2, rows.count { it.state == PhotoMetadataState.UNAVAILABLE })
        assertEquals(10, rows.count { it.state == PhotoMetadataState.PENDING })
        Mockito.verify(storage, Mockito.times(2)).metadata(Mockito.anyString())
    }

    @Test
    fun `busy watermark pool defers the entire sweep`() {
        val id = fixture().also(::old)
        val release = CountDownLatch(1)
        val started = CountDownLatch(2)
        try {
            repeat(2) { watermarkPool.execute { started.countDown(); release.await(10, TimeUnit.SECONDS) } }
            assertTrue(started.await(10, TimeUnit.SECONDS))
            watermarkPool.execute { }
            service.recoverDue()
            assertEquals(PhotoMetadataState.PENDING, metadata.findById(id).orElseThrow().state)
            Mockito.verifyNoInteractions(storage)
        } finally { release.countDown() }
    }

    @Test
    fun `parse suspends an ambient transaction and persists normalized fields independently`() {
        val id = fixture()
        Mockito.`when`(extractor.extract(anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            PhotoMetadataFields(cameraModel = "R6", iso = 400)
        }
        transactions.executeWithoutResult { status ->
            service.extractIfDue(id, byteArrayOf(1))
            status.setRollbackOnly()
        }
        val row = metadata.findById(id).orElseThrow()
        assertEquals(PhotoMetadataState.READY, row.state)
        assertEquals("R6", row.fields?.cameraModel)
        assertEquals(400L, row.fields?.iso)
        assertEquals("camera.jpg", row.originalFilename)
        assertEquals(42L, row.originalSizeBytes)
        assertNotNull(row.extractedAt)
        assertNull(row.nextAttemptAt)
        assertEquals(PhotoStatus.LIVE, photos.findById(id).orElseThrow().status)
    }

    @Test
    fun `empty tags settle EMPTY and fatal parser failure is terminal without changing LIVE`() {
        val empty = fixture()
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields())
        service.extractIfDue(empty, byteArrayOf(1))
        assertEquals(PhotoMetadataState.EMPTY, metadata.findById(empty).orElseThrow().state)
        val failed = fixture()
        Mockito.`when`(extractor.extract(anyArg())).thenThrow(IllegalArgumentException("corrupt tags"))
        service.extractIfDue(failed, byteArrayOf(1))
        service.extractIfDue(failed, byteArrayOf(1))
        val row = metadata.findById(failed).orElseThrow()
        assertEquals(PhotoMetadataState.FAILED, row.state)
        assertEquals(1, row.attempts)
        assertNull(row.nextAttemptAt)
        assertNull(row.leaseToken)
        assertNull(row.fields)
        assertEquals(PhotoStatus.LIVE, photos.findById(failed).orElseThrow().status)
    }
}
