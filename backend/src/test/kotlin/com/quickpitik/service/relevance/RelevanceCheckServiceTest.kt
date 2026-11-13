package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.PhotoScoresRow
import com.quickpitik.repository.PhotoToMeasure
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.SamplingCase
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.support.testTransactionTemplate
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

class RelevanceCheckServiceTest {
    private val batches = Mockito.mock(BatchRelevanceRepository::class.java)
    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val measurer = Mockito.mock(RelevanceMeasurer::class.java)
    private val registry = SimpleMeterRegistry()
    private val clock = Clock.fixed(Instant.parse("2002-03-04T05:06:07Z"), ZoneOffset.UTC)
    private val now = OffsetDateTime.ofInstant(clock.instant(), ZoneOffset.UTC)
    @Suppress("UNCHECKED_CAST")
    private val rekognition = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
        .also { Mockito.`when`(it.getIfAvailable()).thenReturn(Mockito.mock(RekognitionAiClient::class.java)) }
    private val cutoffs = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0)

    private fun service(
        mode: RelevanceMode = RelevanceMode.RECORD_ONLY,
        aiEnabled: Boolean = true,
        junk: RelevanceProperties.Junk = cutoffs,
    ): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = junk)
        val switch = RelevanceSwitch(props, AdminProperties(), AiApiProperties(enabled = aiEnabled), rekognition)
        return RelevanceCheckService(batches, cases, props, switch, clock, testTransactionTemplate(), registry, measurer)
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
        Mockito.verifyNoInteractions(batches, cases, measurer)
    }

    @Test
    fun `with AI off no check starts`() {
        service(aiEnabled = false).tick()
        Mockito.verifyNoInteractions(batches, cases)
    }

    @Test
    fun `with AI off nothing is measured`() {
        sampling()
        service(aiEnabled = false).tick()
        Mockito.verifyNoInteractions(measurer)
        Mockito.verify(cases, Mockito.never()).findSampling(anyArg(), Mockito.anyInt())
    }

    @Test
    fun `photos still to measure go to the measurer and the case stays in sampling`() {
        val c = sampling()
        val todo = listOf(toMeasure(), toMeasure())
        Mockito.`when`(batches.nextToMeasure(c.id, 60)).thenReturn(todo)

        service().tick()

        Mockito.verify(measurer).measure(eqArg(c.batchId), anyArg(), eqArg(todo))
        verifyNotFinished()
        Mockito.verify(batches).releaseCheck(eqArg(c.batchId), anyArg())
    }

    @Test
    fun `a photo measured by an earlier check is not measured again`() {
        // Every sampled photo already has its scores, from this check or an earlier one: nothing is left to measure.
        val c = sampling()
        scores(List(40) { fine() })

        service().tick()

        Mockito.verifyNoInteractions(measurer)
        Mockito.verify(cases).finishSampling(eqArg(c.id), anyArg(), Mockito.eq(40), Mockito.eq(40), Mockito.eq(0), Mockito.eq(0),
            Mockito.eq(0), Mockito.eq(0), eqArg("UNDER"))
    }

    @Test
    fun `a finished sample stores its counts and band and moves to deciding`() {
        val c = sampling()
        scores(List(6) { fine() } + List(2) { fine(fgSharpness = 1.0) } + fine(brightness = 1.0) + fine(personConf = 0.1, raceConf = 0.1))

        service().tick()

        // 4 of 10 are junk: between the two lines.
        Mockito.verify(cases).finishSampling(eqArg(c.id), anyArg(), Mockito.eq(10), Mockito.eq(10), Mockito.eq(2), Mockito.eq(1),
            Mockito.eq(1), Mockito.eq(4), eqArg("MIXED"))
        Mockito.verify(cases, Mockito.never()).startRoundTwo(anyArg(), anyArg())
        Mockito.verify(batches).releaseCheck(eqArg(c.batchId), anyArg())
    }

    @Test
    fun `a sample near the seventy line gets a second round`() {
        val c = sampling()
        scores(List(7) { fine(fgSharpness = 1.0) } + List(3) { fine() })
        Mockito.`when`(cases.startRoundTwo(eqArg(c.id), anyArg())).thenReturn(1)
        Mockito.`when`(cases.pickSample(eqArg(c.id), anyArg(), anyArg(), Mockito.eq(2), Mockito.eq(300))).thenReturn(5)

        service().tick()

        Mockito.verify(cases).pickSample(eqArg(c.id), anyArg(), anyArg(), Mockito.eq(2), Mockito.eq(300))
        verifyNotFinished()
    }

    @Test
    fun `a photo that keeps failing is given up after five tries and the sample can finish`() {
        // The measurer marks it GAVE_UP at the fifth try, so it is no longer to measure; the share leaves it out.
        val c = sampling()
        scores(List(39) { fine() } + fine(measureError = "GAVE_UP"))

        service().tick()

        Mockito.verify(cases).finishSampling(eqArg(c.id), anyArg(), Mockito.eq(40), Mockito.eq(39), Mockito.eq(0), Mockito.eq(0),
            Mockito.eq(0), Mockito.eq(0), eqArg("UNDER"))
    }

    @Test
    fun `a case whose event turned private is closed skipped and nothing is measured`() {
        val c = sampling()
        Mockito.`when`(batches.stillAllowed(c.batchId)).thenReturn(false)

        service().tick()

        Mockito.verify(cases).closeSkipped(eqArg(c.id), anyArg(), eqArg(now))
        Mockito.verifyNoInteractions(measurer)
        Mockito.verify(batches, Mockito.never()).nextToMeasure(anyArg(), Mockito.anyInt())
        verifyNotFinished()
        Mockito.verify(batches).releaseCheck(eqArg(c.batchId), anyArg())
    }

    @Test
    fun `a copy that finds the stage moved on measures nothing`() {
        val c = sampling()
        c.stage = "DECIDING"

        service().tick()

        Mockito.verifyNoInteractions(measurer)
        Mockito.verify(batches, Mockito.never()).nextToMeasure(anyArg(), Mockito.anyInt())
        verifyNotFinished()
        Mockito.verify(batches).releaseCheck(eqArg(c.batchId), anyArg())
    }

    @Test
    fun `a sampling case whose claim is lost is left alone`() {
        val c = sampling()
        Mockito.`when`(batches.claimCheck(eqArg(c.batchId), anyArg(), anyArg(), anyArg())).thenReturn(0)

        service().tick()

        Mockito.verify(cases, Mockito.never()).findById(anyArg())
        Mockito.verifyNoInteractions(measurer)
        Mockito.verify(batches, Mockito.never()).releaseCheck(anyArg(), anyArg())
    }

    // One SAMPLING case of a 40-photo batch, listed, claimable, still allowed, nothing left to measure.
    private fun sampling(): RelevanceCase {
        val c = RelevanceCase(batchId = UUID.randomUUID(), trigger = "QUIET", uploadAt = now, batchPhotos = 40)
        val row = object : SamplingCase {
            override val caseId = c.id
            override val batchId = c.batchId
            override val eventId: UUID = UUID.randomUUID()
            override val photographerId: UUID = UUID.randomUUID()
        }
        Mockito.`when`(cases.findSampling(anyArg(), Mockito.anyInt())).thenReturn(listOf(row))
        Mockito.`when`(batches.claimCheck(eqArg(c.batchId), anyArg(), anyArg(), anyArg())).thenReturn(1)
        Mockito.`when`(cases.findById(c.id)).thenReturn(Optional.of(c))
        Mockito.`when`(batches.stillAllowed(c.batchId)).thenReturn(true)
        return c
    }

    private fun toMeasure(): PhotoToMeasure = object : PhotoToMeasure {
        override val photoId: UUID = UUID.randomUUID()
        override val s3Key = "k/$photoId"
    }

    private fun fine(
        fgSharpness: Double? = 80.0,
        brightness: Double? = 50.0,
        personConf: Double? = 0.99,
        raceConf: Double? = 0.9,
        measureError: String? = null,
    ): PhotoScoresRow = object : PhotoScoresRow {
        override val fgSharpness = fgSharpness
        override val fgBrightness = 60.0
        override val sharpness = 70.0
        override val brightness = brightness
        override val personConf = personConf
        override val raceConf = raceConf
        override val measureError = measureError
    }

    private fun scores(rows: List<PhotoScoresRow>) {
        Mockito.`when`(batches.sampleScores(anyArg())).thenReturn(rows)
    }

    private fun verifyNotFinished() = Mockito.verify(cases, Mockito.never()).finishSampling(anyArg(), anyArg(), Mockito.anyInt(),
        Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(), anyArg())

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
        noCaseWorkBeyondTheSamplingList()
        Mockito.verify(batches, Mockito.never()).releaseCheck(anyArg(), anyArg())
    }

    @Test
    fun `a stale due list after another copy started the check starts nothing`() {
        val row = due().also { listed(it); startable(it) }
        Mockito.`when`(batches.findCheckDueFor(eqArg(row.batchId!!), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(null)

        service().tick()

        noCaseWorkBeyondTheSamplingList()
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

    // The tick always lists sampling cases first; nothing else may touch the cases.
    private fun noCaseWorkBeyondTheSamplingList() {
        Mockito.verify(cases).findSampling(anyArg(), Mockito.anyInt())
        Mockito.verifyNoMoreInteractions(cases)
    }

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
