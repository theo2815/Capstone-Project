package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.repository.BatchCounts
import com.quickpitik.repository.BatchMetadataRow
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.BibOverlap
import com.quickpitik.repository.DueBatch
import com.quickpitik.support.testTransactionTemplate
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class BatchRelevanceServiceTest {
    private val repository = Mockito.mock(BatchRelevanceRepository::class.java)
    private val registry = SimpleMeterRegistry()
    private val clock = Clock.fixed(Instant.parse("2001-06-10T02:00:00Z"), ZoneOffset.UTC)
    private val today = LocalDate.of(2001, 6, 10)
    private val batches = mutableListOf<DueBatch>()
    private val rows = mutableMapOf<UUID, BatchRelevance>()
    private val pending = mutableMapOf<UUID, Long>()
    private val lookups = mutableListOf<Pair<UUID, Int>>()
    private val stamped = mutableListOf<UUID>()


    private fun service(props: RelevanceProperties = RelevanceProperties(mode = RelevanceMode.SHADOW), flagsEnabled: Boolean = true) =
        BatchRelevanceService(repository, props, clock, testTransactionTemplate(), registry,
            AdminProperties(flagsEnabled = flagsEnabled))

    private fun batch(shared: Boolean = true, date: LocalDate = today): DueBatch {
        val b = object : DueBatch {
            override val eventId = UUID.randomUUID()
            override val photographerId = UUID.randomUUID()
            override val lastUploadAt = Instant.parse("2001-06-10T01:00:00.123456Z")
            override val eventDate = date
            override val sharedRace = shared
        }
        val row = BatchRelevance(eventId = b.eventId, photographerId = b.photographerId)
        batches += b
        rows[b.eventId] = row
        Mockito.`when`(repository.findByEventIdAndPhotographerId(b.eventId, b.photographerId)).thenReturn(row)
        Mockito.`when`(repository.lockById(row.id)).thenReturn(row)
        Mockito.`when`(repository.save(row)).thenReturn(row)
        Mockito.`when`(repository.batchCounts(b.eventId, b.photographerId, 4, 8)).thenReturn(counts())
        Mockito.`when`(repository.countPendingLookups(b.eventId, b.photographerId)).thenAnswer { pending[b.eventId] ?: 0L }
        Mockito.`when`(repository.touchSwept(eqArg(row.id), anyArg())).thenAnswer { stamped += b.eventId; 1 }
        Mockito.`when`(repository.lookUpNext(eqArg(row.id), eqArg(b.eventId), eqArg(b.photographerId), Mockito.anyInt(), Mockito.eq(8), anyArg()))
            .thenAnswer { inv ->
                assertTrue(b.eventId in stamped, "queue stamp must commit before the look-up")
                val limit = inv.getArgument<Int>(3)
                lookups += b.eventId to limit
                val inserted = minOf(limit.toLong(), pending[b.eventId] ?: 0L).toInt()
                pending[b.eventId] = (pending[b.eventId] ?: 0L) - inserted
                inserted
            }
        return b
    }

    private fun due() {
        Mockito.`when`(repository.findDueBatches(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(batches)
    }

    private fun counts(tight: Long = 0, looked: Long = 0, processing: Long = 0, indexed: Long = 0, exact: Long = 0) = object : BatchCounts {
        override val photosLive = 60L
        override val photosProcessing = processing
        override val indexedPhotos = indexed
        override val lookups = looked
        override val tightHits = tight
        override val weakHits = 2L
        override val exactMatches = exact
    }

    private fun setCounts(b: DueBatch, c: BatchCounts) {
        Mockito.`when`(repository.batchCounts(b.eventId, b.photographerId, 4, 8)).thenReturn(c)
    }

    private fun metadata(b: DueBatch, dated: Int, outside: Int, pending: Int = 0) {
        val result = (0 until dated + pending).map { i -> object : BatchMetadataRow {
            override val photoId = UUID.randomUUID()
            override val state = if (i >= dated) PhotoMetadataState.PROCESSING else PhotoMetadataState.READY
            override val fields = if (i >= dated) null else PhotoMetadataFields(
                takenAtLocal = b.eventDate.minusDays(if (i < outside) 3 else 0).atTime(6, 0), takenAtOffset = "+08:00",
            )
        } }
        Mockito.`when`(repository.metadataRows(b.eventId, b.photographerId)).thenReturn(result)
    }

    @Test
    fun `mode off touches nothing`() {
        service(RelevanceProperties()).sweep()
        Mockito.verifyNoInteractions(repository)
        assertTrue(registry.meters.isEmpty())
    }

    @Test
    fun `the purge runs on a tick where no batch is due`() {
        due() // empty
        service().sweep()
        // settle-days (7) before the clock, at most max-batches-per-tick (5) batches, words under 3%.
        Mockito.verify(repository).purgeRareWords(3, OffsetDateTime.parse("2001-06-03T02:00Z"), 5,
            OffsetDateTime.parse("2001-06-10T02:00Z"))
    }

    @Test
    fun `a failed purge does not stop the sweep`() {
        Mockito.`when`(repository.purgeRareWords(Mockito.anyInt(), anyArg(), Mockito.anyInt(), anyArg())).thenThrow(RuntimeException("db"))
        due()
        service().sweep()
        Mockito.verify(repository).findDueBatches(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())
    }

    @Test
    fun `a tick spends at most the look-up budget across batches`() {
        val a = batch(); val b = batch(); val c = batch()
        pending[a.eventId] = 60; pending[b.eventId] = 60; pending[c.eventId] = 10
        due()
        service().sweep()
        assertEquals(listOf(a.eventId to 100, b.eventId to 40), lookups)
        assertEquals(20L, pending[b.eventId])
        assertNull(rows.getValue(b.eventId).scoredAt)
        assertNull(rows.getValue(c.eventId).scoredAt)
    }

    @Test
    fun `a batch given no budget keeps its place and one given budget is stamped before its look-up`() {
        val a = batch(); val b = batch(); val covered = batch()
        pending[a.eventId] = 100; pending[b.eventId] = 1
        due()
        service().sweep()
        assertEquals(listOf(a.eventId, covered.eventId), stamped)
        assertNotNull(rows.getValue(covered.eventId).scoredAt)
    }

    @Test
    fun `scoring locks the batch row before it reads`() {
        val b = batch()
        val row = rows.getValue(b.eventId)
        Mockito.`when`(repository.findByEventIdAndPhotographerId(b.eventId, b.photographerId)).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive(), "the id lookup must not prime the scoring persistence context")
            row
        }
        Mockito.`when`(repository.lockById(row.id)).thenAnswer {
            assertTrue(TransactionSynchronizationManager.isActualTransactionActive(), "the row lock belongs to the scoring transaction")
            row
        }
        service().score(b, today)
        val order = Mockito.inOrder(repository)
        order.verify(repository).lockById(rows.getValue(b.eventId).id)
        order.verify(repository).batchCounts(b.eventId, b.photographerId, 4, 8)
    }

    @Test
    fun `a batch with photos still pending is not scored`() {
        val b = batch(); pending[b.eventId] = 101; due()
        service().sweep()
        assertNull(rows.getValue(b.eventId).scoredAt)
        Mockito.verify(repository, Mockito.never()).lockById(rows.getValue(b.eventId).id)
    }

    @Test
    fun `a covered batch is scored with the upload time the due query saw`() {
        val b = batch(); due()
        service().sweep()
        assertEquals(b.lastUploadAt, rows.getValue(b.eventId).scoredUploadAt?.toInstant())
        assertEquals(today, rows.getValue(b.eventId).scoredEventDate)
    }

    @Test
    fun `three tight hits in sixty look-ups scores low by the copy rule`() {
        val b = batch(); setCounts(b, counts(tight = 3, looked = 60))
        service().score(b, today)
        assertEquals("LOW", rows.getValue(b.eventId).level)
        assertEquals(true, rows.getValue(b.eventId).copyLow)
    }

    @Test
    fun `a shared race with nine of ten dated photos outside scores low by the date rule`() {
        val b = batch(); metadata(b, 10, 9)
        service().score(b, today)
        val row = rows.getValue(b.eventId)
        assertEquals("LOW", row.level)
        assertEquals(10, row.datedPhotos)
        assertEquals(9, row.datedOutside)
        assertEquals(true, row.dateLow)
    }

    @Test
    fun `pending metadata leaves the date rule undecided and the batch not final`() {
        val b = batch(date = today.minusDays(5)); metadata(b, 10, 10, 1)
        service().score(b, today)
        val row = rows.getValue(b.eventId)
        assertEquals(1, row.metadataPending)
        assertNull(row.dateLow)
        assertNull(row.level)
        assertNull(row.finalScoredAt)
    }

    @Test
    fun `a photographer's own event gets no date counts`() {
        val b = batch(shared = false)
        service().score(b, today)
        val row = rows.getValue(b.eventId)
        assertNull(row.datedPhotos); assertNull(row.datedOutside); assertNull(row.metadataPending); assertNull(row.dateLow)
        Mockito.verify(repository, Mockito.never()).metadataRows(b.eventId, b.photographerId)
    }

    @Test
    fun `no indexed photos leaves bib evidence empty`() {
        val b = batch()
        service().score(b, today)
        assertNull(rows.getValue(b.eventId).bibbedPhotos)
        assertNull(rows.getValue(b.eventId).bibOverlapPhotos)
        Mockito.verify(repository, Mockito.never()).bibOverlap(b.eventId, b.photographerId)
    }

    @Test
    fun `exact and xmp matches are stored and never set the level`() {
        val b = batch(); setCounts(b, counts(looked = 60, indexed = 10, exact = 8))
        Mockito.`when`(repository.xmpMatches(b.eventId, b.photographerId)).thenReturn(7)
        Mockito.`when`(repository.bibOverlap(b.eventId, b.photographerId)).thenReturn(object : BibOverlap {
            override val bibbedPhotos = 9L
            override val bibOverlapPhotos = 6L
        })
        service().score(b, today)
        val row = rows.getValue(b.eventId)
        assertEquals(8, row.exactMatches); assertEquals(7, row.xmpMatches)
        assertEquals(9, row.bibbedPhotos); assertEquals(6, row.bibOverlapPhotos)
        assertNull(row.level)
    }

    @Test
    fun `a closed event with nothing pending gets its final scoring stamp`() {
        val b = batch(date = today.minusDays(4))
        service().score(b, today)
        assertEquals(OffsetDateTime.now(clock), rows.getValue(b.eventId).finalScoredAt)
    }

    @Test
    fun `a closed event still pending after the settle days is stamped anyway`() {
        val b = batch(date = today.minusDays(11)); metadata(b, 10, 10, 1)
        setCounts(b, counts(processing = 1))
        service().score(b, today)
        assertNotNull(rows.getValue(b.eventId).finalScoredAt)
        assertEquals(true, rows.getValue(b.eventId).dateLow)
        assertEquals("LOW", rows.getValue(b.eventId).level)
        assertEquals(1, rows.getValue(b.eventId).metadataPending)
    }

    @Test
    fun `at the settle deadline the date rule still waits for metadata`() {
        val b = batch(date = today.minusDays(10)); metadata(b, 10, 10, 1)
        service().score(b, today)
        assertNull(rows.getValue(b.eventId).dateLow)
        assertNull(rows.getValue(b.eventId).finalScoredAt)
    }

    @Test
    fun `one failing batch does not stop the others`() {
        val a = batch(); val b = batch(); due()
        pending[a.eventId] = 1
        Mockito.`when`(repository.setLocalStatementTimeout(5000)).thenThrow(RuntimeException("statement timeout"))
        service().sweep()
        assertTrue(a.eventId in stamped)
        assertNull(rows.getValue(a.eventId).scoredAt)
        assertNotNull(rows.getValue(b.eventId).scoredAt)
    }

    @Test
    fun `the batch log line carries every count`() {
        val row = BatchRelevance(eventId = UUID(0, 1), photographerId = UUID(0, 2),
            copyLow = true, dateLow = null, lookups = 60, tightHits = 3, weakHits = 2,
            exactMatches = 4, xmpMatches = 5, datedPhotos = 10, datedOutside = 8,
            metadataPending = 1, bibbedPhotos = 7, bibOverlapPhotos = 6, level = "LOW")
        assertEquals("Relevance batch event=00000000-0000-0000-0000-000000000001 photographer=00000000-0000-0000-0000-000000000002 level=low copyLow=true dateLow=null lookups=60 tight=3 weak=2 exact=4 xmp=5 dated=10 outside=8 pendingMeta=1 bibbed=7 bibOverlap=6 final=false", formatRelevanceBatchLine(row))
        row.level = null
        assertTrue(formatRelevanceBatchLine(row).contains("level=none"))
    }

    @Test
    fun `look-ups and scored batches are counted`() {
        val b = batch(); pending[b.eventId] = 3; setCounts(b, counts(tight = 3, looked = 3)); due()
        service().sweep()
        assertEquals(3.0, registry.get("qp.relevance.lookups").counter().count())
        assertEquals(1.0, registry.get("qp.relevance.scored").tag("level", "low").counter().count())
        assertEquals(1L, registry.get("qp.relevance.tick").timer().count())
        Mockito.verify(repository).setLocalStatementTimeout(5000)
    }

    private inline fun <reified T : Any> anyArg(): T = Mockito.any(T::class.java) ?: uninitialized()
    private fun <T : Any> eqArg(value: T): T = Mockito.eq(value) ?: value
    @Suppress("UNCHECKED_CAST")
    private fun <T> uninitialized(): T = null as T
}
