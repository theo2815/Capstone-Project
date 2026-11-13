package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.PhotoMetadata
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.DueBatch
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.relevance.BatchRelevanceService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class BatchRelevanceSweepIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var metadata: PhotoMetadataRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var systemFlags: SystemFlagService
    @Autowired private lateinit var flags: FlagRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService
    @Autowired private lateinit var events: EventRepository

    private fun service(mode: RelevanceMode = RelevanceMode.SHADOW) = BatchRelevanceService(batches,
        RelevanceProperties(mode = mode, maxBatchesPerTick = 10_000),
        Clock.fixed(NOW.toInstant(), ZoneOffset.UTC), tx, SimpleMeterRegistry(),
        systemFlags, flags, users, AdminProperties(flagsEnabled = true), decisions, events)

    // ---- PE-2.1 flag mode ----

    @Test
    fun `in flag mode a low batch opens one flag and a second tick opens no second one`() {
        val (original, copies) = copiedBatches()
        val s = service(RelevanceMode.FLAG)
        s.sweep(); s.sweep(); s.sweep()
        val flag = relevanceFlags(copies.photographerId).single()
        assertEquals(flag, row(copies).flagId)
        assertNotNull(row(copies).flaggedAt)
        assertEquals(emptyList(), relevanceFlags(original.photographerId))
    }

    @Test
    fun `a batch that went low in shadow mode is flagged once flag mode is on`() {
        val (_, copies) = copiedBatches()
        service().sweep(); service().sweep()
        assertEquals("LOW", row(copies).level)
        assertNull(row(copies).flagId)
        assertEquals(emptyList(), relevanceFlags(copies.photographerId))

        service(RelevanceMode.FLAG).sweep()

        assertEquals(relevanceFlags(copies.photographerId).single(), row(copies).flagId)
    }

    @Test
    fun `two threads scoring two low batches of one photographer open one flag`() {
        val (_, first) = copiedBatches()
        val (_, second) = copiedBatches(copier = first.photographerId)
        service().sweep(); service().sweep()
        assertEquals(listOf("LOW", "LOW"), listOf(row(first).level, row(second).level))

        val s = service(RelevanceMode.FLAG)
        val barrier = CyclicBarrier(2)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val runs = listOf(first, second).map { b -> pool.submit<Unit> { barrier.await(10, TimeUnit.SECONDS); s.score(b, TODAY) } }
            runs.forEach { it.get(30, TimeUnit.SECONDS) }
        } finally {
            pool.shutdownNow()
        }

        val flag = relevanceFlags(first.photographerId).single()
        assertEquals(listOf(flag, flag), listOf(row(first).flagId, row(second).flagId))
        assertEquals(2, jdbc.queryForObject("SELECT note FROM flags WHERE id = ?", String::class.java, flag)!!.lines().size)
    }

    private fun relevanceFlags(photographer: UUID): List<UUID> = jdbc.queryForList(
        "SELECT id FROM flags WHERE target_kind = 'user' AND target_id = ? AND reason = 'batch_relevance_low'", UUID::class.java, photographer)

    @Test
    fun `a copied batch ends low and the original photographer's batch does not`() {
        val (original, copies) = copiedBatches()
        service().sweep()
        service().sweep()
        assertNull(row(original).level)
        val result = row(copies)
        assertEquals("LOW", result.level)
        assertEquals(5, result.tightHits)
        assertEquals(40, result.lookups)
    }

    @Test
    fun `trashing the copies and scoring again clears the level`() {
        val (_, copies) = copiedBatches()
        val s = service()
        s.sweep(); s.sweep()
        assertEquals("LOW", row(copies).level)
        jdbc.update("UPDATE photos SET deleted_at = ? WHERE event_id = ? AND photographer_id = ? AND phash_clean = 1", NOW, copies.eventId, copies.photographerId)
        // A trash does not make a batch due; only this explicit score reads the changed evidence.
        s.score(copies, TODAY)
        assertNull(row(copies).level)
        assertEquals(0, row(copies).tightHits)
        assertEquals(35, row(copies).lookups)
    }

    @Test
    fun `a corrected event date clears a date rule low at the next tick`() {
        val b = batch(TODAY.minusDays(10))
        repeat(10) {
            metadata.saveAndFlush(PhotoMetadata(photoId = photo(b, null), state = PhotoMetadataState.READY,
                fields = PhotoMetadataFields(takenAtLocal = TODAY.atTime(6, 0), takenAtOffset = "+08:00")))
        }
        val s = service()
        s.sweep()
        assertEquals("LOW", row(b).level)
        assertNotNull(row(b).finalScoredAt)
        jdbc.update("UPDATE events SET date = ? WHERE id = ?", TODAY, b.eventId)
        s.sweep()
        assertNull(row(b).level)
        assertEquals(TODAY, row(b).scoredEventDate)
        assertNull(row(b).finalScoredAt)
    }

    @Test
    fun `running the tick twice changes nothing the second time`() {
        val b = batch()
        photo(b, 1)
        val s = service()
        s.sweep()
        val first = row(b)
        val originalStamp = first.lastSweptAt
        assertEquals(UPLOAD.toInstant(), first.scoredUploadAt?.toInstant())
        // A sentinel makes a second stamp observable even though the service clock is fixed.
        jdbc.update("UPDATE batch_relevance SET last_swept_at = ? WHERE id = ?", NOW.minusMinutes(1), first.id)
        s.sweep()
        val second = row(b)
        assertEquals(NOW.minusMinutes(1).toInstant(), second.lastSweptAt?.toInstant())
        assertEquals(first.scoredAt, second.scoredAt)
        assertEquals(1, second.lookups)
        assertNotNull(originalStamp)
    }

    private fun copiedBatches(copier: UUID? = null): Pair<DueBatch, DueBatch> {
        val original = batch()
        val copies = batch(event = original.eventId, upload = UPLOAD.plusMinutes(1), photographerId = copier)
        photo(original, 0, UPLOAD)
        repeat(40) { photo(copies, if (it < 5) 1 else -1, UPLOAD.plusMinutes(1)) }
        return original to copies
    }

    private fun batch(date: LocalDate = TODAY, event: UUID? = null, upload: OffsetDateTime = UPLOAD, photographerId: UUID? = null): DueBatch {
        val photographer = photographerId
            ?: id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')", "rel-${UUID.randomUUID()}@test.local")
        val eventId = event ?: id("INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Rel Run', ?, 'Cebu', 'ACTIVE')", "rel-${UUID.randomUUID()}", date)
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?)", eventId, photographer, upload, upload)
        return object : DueBatch {
            override val eventId = eventId
            override val photographerId = photographer
            override val lastUploadAt = upload.toInstant()
            override val eventDate = date
            override val sharedRace = true
        }
    }

    private fun photo(b: DueBatch, hash: Long?, upload: OffsetDateTime = UPLOAD): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, uploaded_at, status, phash_clean) VALUES (?, ?, ?, 125, ?, 'LIVE', ?)",
        b.eventId, b.photographerId, "rel/${UUID.randomUUID()}", upload, hash)

    private fun row(b: DueBatch): BatchRelevance = assertNotNull(
        batches.findByEventIdAndPhotographerId(b.eventId, b.photographerId), "the sweep must create and score this batch")
    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        // R2-76: the due-query fixtures use 2001 (ordering case 1998); this sweep uses 1999.
        val NOW: OffsetDateTime = OffsetDateTime.parse("1999-06-10T02:00:00.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(30)
        val TODAY: LocalDate = LocalDate.of(1999, 6, 10)
    }
}
