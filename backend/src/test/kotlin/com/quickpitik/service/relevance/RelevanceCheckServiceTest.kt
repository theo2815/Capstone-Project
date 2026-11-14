package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.Event
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoScoresRow
import com.quickpitik.repository.PhotoToMeasure
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.SamplingCase
import com.quickpitik.repository.WordCount
import com.quickpitik.repository.WordPhotographers
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.websocket.AdminInboxEvent
import com.quickpitik.support.testTransactionTemplate
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.context.ApplicationEventPublisher
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class RelevanceCheckServiceTest {
    private val batches = Mockito.mock(BatchRelevanceRepository::class.java)
    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val measurer = Mockito.mock(RelevanceMeasurer::class.java)
    private val client = Mockito.mock(AnthropicClient::class.java)
    private val events = Mockito.mock(EventRepository::class.java)
    private val alerts = Mockito.mock(RelevanceAlertRepository::class.java)
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
    private val holds = Mockito.mock(com.quickpitik.repository.RelevanceHoldRepository::class.java)
    private val decisionLog = Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java)
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
        key: String = "",
    ): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = junk, llm = RelevanceProperties.Llm(apiKey = key))
        val switch = RelevanceSwitch(props, AdminProperties(), AiApiProperties(enabled = aiEnabled), rekognition)
        return RelevanceCheckService(batches, cases, props, switch, clock, testTransactionTemplate(), registry, measurer,
            client, events, alerts, publisher, LegalProperties(currentVersion = "1.5"),
            holds, Mockito.mock(com.quickpitik.repository.PhotoRepository::class.java), decisionLog)
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
        Mockito.verifyNoInteractions(batches)
        // Only the apply step, which reads the database and runs with AI off (G15).
        Mockito.verify(cases).findDecided(Mockito.anyInt())
        Mockito.verifyNoMoreInteractions(cases)
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

    // The tick always lists decided and sampling cases first; nothing else may touch the cases.
    private fun noCaseWorkBeyondTheSamplingList() {
        Mockito.verify(cases).findDecided(Mockito.anyInt())
        Mockito.verify(cases).findSampling(anyArg(), Mockito.anyInt())
        Mockito.verifyNoMoreInteractions(cases)
    }

    // RV-1.8b: the decide step.

    @Test
    fun `a failed call is tried again next tick and becomes can't tell after three`() {
        val c = deciding()
        answers(LlmResult.Failed("http 529 after 3 tries"))
        val decider = service(key = "k")

        decider.tick()
        c.decideAttempts = 1
        decider.tick()
        assertTrue(saved.isEmpty(), "two failed calls: still deciding")
        c.decideAttempts = 2
        decider.tick()

        Mockito.verify(client, Mockito.times(3)).ask(anyArg(), anyArg(), Mockito.anyInt())
        Mockito.verify(cases, Mockito.times(3)).raiseDecideAttempts(eqArg(c.id), anyArg())
        assertEquals(listOf("CANT_TELL", "CALL_FAILED", null), saved().let { listOf(it["verdict"], it["note"], it["modelVerdict"]) })
        assertEquals(1.0, registry.counter("qp.relevance.decided", "verdict", "cant_tell").count())
    }

    @Test
    fun `an answer that contradicts the band is stored as can't tell with what the model said`() {
        deciding(band = "UNDER")
        answers(answer("junk_dump", reason = "Most photos are black."))

        service(key = "k").tick()

        val row = saved()
        assertEquals(listOf("CANT_TELL", "CONTRADICTS", "JUNK_DUMP"), listOf(row["verdict"], row["note"], row["modelVerdict"]))
        assertEquals("Most photos are black.", row["adminReason"])
        assertEquals("junk_share", row["evidenceUsed"])
    }

    @Test
    fun `a 200-character verdict from the model does not break the write`() {
        deciding()
        answers(answer("x".repeat(200), model = "m".repeat(200)))

        service(key = "k").tick()

        val row = saved()
        assertEquals(listOf("CANT_TELL", "UNKNOWN_VALUE", null), listOf(row["verdict"], row["note"], row["modelVerdict"]))
        assertEquals(60, (row["model"] as String).length, "the model column holds 60")
    }

    @Test
    fun `the summary, the model and the prompt version are stored`() {
        val c = deciding()
        Mockito.`when`(batches.countWordPhotos(c.batchId)).thenReturn(30L)
        Mockito.`when`(batches.topWords(c.batchId, 3, 30)).thenReturn(listOf(word("CEBU", 25)))
        Mockito.`when`(batches.otherPhotographers(eventId, c.batchId)).thenReturn(2)
        Mockito.`when`(batches.eventWords(eqArg(eventId), eqArg(c.batchId), anyArg(), Mockito.eq(3))).thenReturn(listOf(photographers("CEBU", 2)))
        answers(answer("fits", model = "claude-sonnet-5-5-20261001"))

        service(key = "k").tick()

        val sent = ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(client).ask(eqArg(PROMPT_RV1), sent.capture() ?: "", Mockito.eq(4096))
        val row = saved()
        assertEquals(sent.value, row["summary"])
        assertTrue(sent.value.startsWith("""{"case":"${c.id}","event":{"name":"Rel Run","date":"2002-03-03","place":"Cebu","shared_race":true}"""), sent.value)
        assertTrue(""""copy_rule":"clear","date_rule":"clear","words_from_photos":30,"batch_words":[{"w":"CEBU","photos":25}],""" +
            """"other_photographers":2,"event_words":[{"w":"CEBU","photographers":2}]}""" in sent.value, sent.value)
        assertFalse(photographerId.toString() in sent.value, "no photographer id")
        assertEquals(listOf("FITS", null, "FITS", "claude-sonnet-5-5-20261001", "rv1", false, false, 30),
            listOf("verdict", "note", "modelVerdict", "model", "promptVersion", "copyLow", "dateLow", "wordsPhotos").map { row[it] })
        Mockito.verify(batches).releaseCheck(eqArg(c.batchId), anyArg())
    }

    @Test
    fun `a case decided before version 1 scored its upload sends not finished and keeps no rule result`() {
        val c = deciding(createdAt = now.minusMinutes(31))
        batch.scoredUploadAt = c.uploadAt.minusHours(1)
        batch.copyLow = true
        answers(answer("copied"))

        service(key = "k").tick()

        val row = saved()
        assertTrue(""""copy_rule":"not_finished","date_rule":"not_run"""" in (row["summary"] as String))
        assertEquals(listOf("CANT_TELL", "CONTRADICTS", null, null), listOf(row["verdict"], row["note"], row["copyLow"], row["dateLow"]))
    }

    @Test
    fun `no transaction is open during the call`() {
        val c = deciding()
        var inTransaction: Boolean? = null
        Mockito.`when`(client.ask(anyArg(), anyArg(), Mockito.anyInt())).thenAnswer {
            inTransaction = TransactionSynchronizationManager.isActualTransactionActive()
            answer("fits")
        }

        service(key = "k").tick()

        assertEquals(false, inTransaction)
        val order = Mockito.inOrder(cases, client)
        order.verify(cases).raiseDecideAttempts(eqArg(c.id), anyArg())
        order.verify(client).ask(anyArg(), anyArg(), Mockito.anyInt())
        assertEquals(1, saved.size)
    }

    @Test
    fun `with no key nothing is decided`() {
        deciding()

        service(key = "").tick()

        Mockito.verify(cases, Mockito.never()).findDeciding(anyArg(), anyArg(), Mockito.anyInt())
        Mockito.verifyNoInteractions(client)
    }

    @Test
    fun `a case waits for version 1's score, but not past the wait time`() {
        val c = deciding(createdAt = now.minusMinutes(29))
        batch.scoredUploadAt = c.uploadAt.minusHours(1)
        answers(answer("fits"))
        val decider = service(key = "k")

        decider.tick()
        Mockito.verify(cases).findDeciding(eqArg(now), eqArg(now.minusMinutes(30)), Mockito.eq(3))
        Mockito.verifyNoInteractions(client)

        batch.scoredUploadAt = c.uploadAt
        decider.tick()
        Mockito.verify(client, Mockito.times(1)).ask(anyArg(), anyArg(), Mockito.anyInt())

        val late = deciding(createdAt = now.minusMinutes(30))
        batch.scoredUploadAt = null
        decider.tick()
        Mockito.verify(client, Mockito.times(2)).ask(anyArg(), anyArg(), Mockito.anyInt())
        assertEquals(late.id, saved.last()[0])
    }

    @Test
    fun `a copy that finds the case already decided makes no call`() {
        val c = deciding()
        c.stage = "DECIDED"
        val decider = service(key = "k")

        decider.tick()
        Mockito.verify(cases, Mockito.never()).raiseDecideAttempts(anyArg(), anyArg())

        // Or the stage moves between the read and the raise.
        c.stage = "DECIDING"
        Mockito.`when`(cases.raiseDecideAttempts(eqArg(c.id), anyArg())).thenReturn(0)
        decider.tick()

        Mockito.verifyNoInteractions(client)
        assertTrue(saved.isEmpty())
        Mockito.verify(batches, Mockito.times(2)).releaseCheck(eqArg(c.batchId), anyArg())
    }

    @Test
    fun `a case whose event turned private is closed skipped and no call is made`() {
        val c = deciding()
        Mockito.`when`(batches.stillAllowed(c.batchId)).thenReturn(false)

        service(key = "k").tick()

        Mockito.verify(cases).closeSkippedDeciding(eqArg(c.id), anyArg(), eqArg(now))
        Mockito.verifyNoInteractions(client)
        Mockito.verify(cases, Mockito.never()).raiseDecideAttempts(anyArg(), anyArg())
        Mockito.verify(batches).releaseCheck(eqArg(c.batchId), anyArg())
    }

    private val eventId: UUID = UUID.randomUUID()
    private val photographerId: UUID = UUID.randomUUID()
    private lateinit var batch: BatchRelevance
    private val saved = mutableListOf<Array<Any?>>()

    // One DECIDING case of a 40-photo batch on a shared race, listed, claimable, still allowed, and scored by
    // version 1 for the same upload (copy and date clear).
    private fun deciding(band: String = "UNDER", createdAt: OffsetDateTime = now.minusMinutes(1)): RelevanceCase {
        val upload = now.minusMinutes(20)
        val c = RelevanceCase(batchId = UUID.randomUUID(), trigger = "QUIET", uploadAt = upload, batchPhotos = 40,
            createdAt = createdAt, stage = "DECIDING", sampleSize = 40, measured = 40, blurry = 1, dark = 0, notRace = 1,
            junk = 2, band = band)
        batch = BatchRelevance(id = c.batchId, eventId = eventId, photographerId = photographerId, scoredAt = now,
            scoredUploadAt = upload, copyLow = false, dateLow = false, datedPhotos = 20)
        val row = object : SamplingCase {
            override val caseId = c.id
            override val batchId = c.batchId
            override val eventId = this@RelevanceCheckServiceTest.eventId
            override val photographerId = this@RelevanceCheckServiceTest.photographerId
        }
        val event = Mockito.mock(Event::class.java)
        Mockito.`when`(event.name).thenReturn("Rel Run")
        Mockito.`when`(event.date).thenReturn(LocalDate.of(2002, 3, 3))
        Mockito.`when`(event.location).thenReturn("Cebu")
        Mockito.`when`(event.createdBy).thenReturn(null)
        Mockito.`when`(cases.findDeciding(anyArg(), anyArg(), Mockito.anyInt())).thenReturn(listOf(row))
        Mockito.`when`(batches.claimCheck(eqArg(c.batchId), anyArg(), anyArg(), anyArg())).thenReturn(1)
        Mockito.`when`(cases.findById(c.id)).thenReturn(Optional.of(c))
        Mockito.`when`(batches.stillAllowed(c.batchId)).thenReturn(true)
        Mockito.`when`(batches.findById(c.batchId)).thenAnswer { Optional.of(batch) }
        Mockito.`when`(events.findById(eventId)).thenReturn(Optional.of(event))
        Mockito.`when`(cases.raiseDecideAttempts(eqArg(c.id), anyArg())).thenReturn(1)
        Mockito.`when`(cases.saveVerdict(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(),
            anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())).thenAnswer { saved += it.arguments; 1 }
        return c
    }

    private fun answers(result: LlmResult) {
        Mockito.`when`(client.ask(anyArg(), anyArg(), Mockito.anyInt())).thenReturn(result)
    }

    private fun answer(verdict: String, reason: String = "Looks fine.", model: String = "claude-sonnet-5-5"): LlmResult =
        LlmResult.Answer("""{"verdict":"$verdict","evidence_used":["junk_share"],"admin_reason":"$reason"}""", model)

    // The one saved verdict, by the names of saveVerdict's parameters.
    private fun saved(): Map<String, Any?> = listOf("id", "token", "now", "verdict", "note", "modelVerdict", "adminReason",
        "evidenceUsed", "summary", "model", "promptVersion", "copyLow", "dateLow", "wordsPhotos").zip(saved.single()).toMap()

    private fun word(w: String, n: Int) = object : WordCount {
        override val word = w
        override val photos = n
    }

    private fun photographers(w: String, n: Int) = object : WordPhotographers {
        override val word = w
        override val photographers = n
    }

    // RV-1.9: the apply step.

    @Test
    fun `the admin ping goes out after the commit, once per new alert`() {
        val c = RelevanceCase(batchId = UUID.randomUUID(), trigger = "QUIET", uploadAt = now, batchPhotos = 600,
            stage = "DECIDED", verdict = "COPIED")
        Mockito.`when`(cases.findDecided(Mockito.anyInt())).thenReturn(listOf(c))
        Mockito.`when`(cases.findById(c.id)).thenReturn(Optional.of(c))
        Mockito.`when`(batches.stillAllowed(c.batchId)).thenReturn(true)
        Mockito.`when`(cases.applyCase(eqArg(c.id), anyArg(), anyArg(), anyArg())).thenReturn(1)
        // The big-hide alert is new; the copied one was already raised by another server copy.
        val ids = mutableMapOf<String, UUID>()
        Mockito.`when`(alerts.raise(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyBoolean(), Mockito.anyInt(), anyArg()))
            .thenAnswer {
                val kind = it.getArgument<String>(3)
                ids[kind] = it.getArgument(0)
                if (kind == "BIG_HIDE") 1 else 0
            }
        val inTransaction = mutableListOf<Boolean>()
        Mockito.`when`(publisher.publishEvent(anyArg<Any>())).thenAnswer {
            // Published inside the transaction, so NotificationBroadcaster's AFTER_COMMIT listener sends it after the commit.
            inTransaction += TransactionSynchronizationManager.isSynchronizationActive()
            null
        }

        service(aiEnabled = false).tick()

        val sent = ArgumentCaptor.forClass(AdminInboxEvent::class.java)
        Mockito.verify(publisher).publishEvent(sent.capture())
        assertEquals(listOf(true), inTransaction)
        assertEquals(mapOf("type" to "relevance_alert", "entityId" to ids["BIG_HIDE"].toString(), "actorId" to c.id.toString()),
            sent.value.payload - "occurredAt")
        assertEquals(setOf("BIG_HIDE", "COPIED"), ids.keys)
    }

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
