package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.support.testTransactionTemplate
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals

class RelevanceCheckServiceTest {
    private val batches = Mockito.mock(BatchRelevanceRepository::class.java)
    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val registry = SimpleMeterRegistry()
    private val clock = Clock.fixed(Instant.parse("2002-03-04T05:06:07Z"), ZoneOffset.UTC)

    private fun service(mode: RelevanceMode = RelevanceMode.RECORD_ONLY, aiEnabled: Boolean = true): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode)
        val switch = RelevanceSwitch(props, AdminProperties(), AiApiProperties(enabled = aiEnabled))
        return RelevanceCheckService(batches, cases, props, switch, clock, testTransactionTemplate(), registry)
    }

    private fun due(batchId: UUID = UUID.randomUUID()): CheckDue = object : CheckDue {
        override val batchId = batchId
        override val eventId = UUID.randomUUID()
        override val photographerId = UUID.randomUUID()
        override val lastUploadAt = Instant.parse("2002-03-04T04:46:07.123456Z")
        override val photoCount = 40
        override val trigger = "QUIET"
    }

    private fun listed(vararg rows: CheckDue) {
        Mockito.`when`(batches.findCheckDue(anyArg(), anyArg(), Mockito.anyInt(), Mockito.anyInt())).thenReturn(rows.toList())
    }

    // A batch whose claim is won and which is still due when read again.
    private fun startable(row: CheckDue) {
        Mockito.`when`(batches.claimCheck(eqArg(row.batchId!!), anyArg(), anyArg(), anyArg())).thenReturn(1)
        Mockito.`when`(batches.findCheckDueFor(eqArg(row.batchId!!), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(row)
        Mockito.`when`(batches.markChecked(eqArg(row.batchId!!), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        Mockito.`when`(cases.saveAndFlush(anyArg<RelevanceCase>())).thenAnswer { it.getArgument<RelevanceCase>(0) }
    }

    @Test
    fun `with the mode off the tick reads nothing`() {
        service(mode = RelevanceMode.OFF).tick()
        Mockito.verifyNoInteractions(batches, cases)
    }

    @Test
    fun `with AI off no check starts`() {
        service(aiEnabled = false).tick()
        Mockito.verifyNoInteractions(batches, cases)
    }

    @Test
    fun `a due batch gets one case, its sample and the checked mark, then the claim is released`() {
        val row = due().also { listed(it); startable(it) }

        service().tick()

        val saved = ArgumentCaptor.forClass(RelevanceCase::class.java)
        Mockito.verify(cases).saveAndFlush(saved.capture())
        assertEquals(row.batchId, saved.value.batchId)
        assertEquals("QUIET", saved.value.trigger)
        assertEquals("SAMPLING", saved.value.stage)
        assertEquals(40, saved.value.batchPhotos)
        assertEquals(row.lastUploadAt, saved.value.uploadAt.toInstant())
        Mockito.verify(cases).pickSample(saved.value.id, row.eventId, row.photographerId, 1, 300)
        Mockito.verify(batches).markChecked(eqArg(row.batchId!!), anyArg(), eqArg(row.lastUploadAt.atOffset(ZoneOffset.UTC)), Mockito.eq(40))
        Mockito.verify(batches).releaseCheck(eqArg(row.batchId!!), anyArg())
        assertEquals(1L, registry.timer("qp.relevance.check.tick").count())
    }

    @Test
    fun `a lost claim starts nothing`() {
        val row = due().also { listed(it) }
        Mockito.`when`(batches.claimCheck(eqArg(row.batchId!!), anyArg(), anyArg(), anyArg())).thenReturn(0)

        service().tick()

        Mockito.verify(batches, Mockito.never()).findCheckDueFor(anyArg(), anyArg(), anyArg(), Mockito.anyInt())
        Mockito.verifyNoInteractions(cases)
        Mockito.verify(batches, Mockito.never()).releaseCheck(anyArg(), anyArg())
    }

    @Test
    fun `a stale due list after another copy started the check starts nothing`() {
        val row = due().also { listed(it); startable(it) }
        Mockito.`when`(batches.findCheckDueFor(eqArg(row.batchId!!), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(null)

        service().tick()

        Mockito.verifyNoInteractions(cases)
        Mockito.verify(batches, Mockito.never()).markChecked(anyArg(), anyArg(), anyArg(), Mockito.anyInt())
        Mockito.verify(batches).releaseCheck(eqArg(row.batchId!!), anyArg())
    }

    @Test
    fun `one failing batch does not stop the next`() {
        val broken = due()
        val fine = due()
        listed(broken, fine)
        startable(fine)
        Mockito.`when`(batches.claimCheck(eqArg(broken.batchId!!), anyArg(), anyArg(), anyArg())).thenThrow(IllegalStateException("boom"))

        service().tick()

        Mockito.verify(cases).pickSample(anyArg(), eqArg(fine.eventId), eqArg(fine.photographerId), Mockito.eq(1), Mockito.eq(300))
        Mockito.verify(batches).releaseCheck(eqArg(fine.batchId!!), anyArg())
    }

    @Test
    fun `a batch with no row yet gets one before the claim`() {
        val id = UUID.randomUUID()
        val noRow = object : CheckDue by due(id) { override val batchId: UUID? = null }
        listed(noRow)
        startable(due(id))
        val row = com.quickpitik.entity.BatchRelevance(id = id, eventId = noRow.eventId, photographerId = noRow.photographerId)
        Mockito.`when`(batches.findByEventIdAndPhotographerId(noRow.eventId, noRow.photographerId)).thenReturn(row)

        service().tick()

        Mockito.verify(batches).ensureBatch(noRow.eventId, noRow.photographerId)
        Mockito.verify(batches).claimCheck(eqArg(id), anyArg(), anyArg(), anyArg())
    }

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
