package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.OwedNote
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.context.ApplicationEventPublisher
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals

// RV-3.1 (A.5): the notify step of the tick. The safety tests and the choice of text; the queries run in
// RelevanceNotifierIntegrationTest.
class RelevanceNotifierTest {
    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val decisionLog = Mockito.mock(AdminDecisionLogService::class.java)
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
    // 2026-11-17 20:00 UTC is 2026-11-18 04:00 in Manila: the appeal date is counted in Philippine time.
    private val clock = Clock.fixed(Instant.parse("2026-11-17T20:00:00Z"), ZoneOffset.UTC)
    @Suppress("UNCHECKED_CAST")
    private val rekognition = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)

    @Test
    fun `with the mode off nothing runs`() {
        notifier(RelevanceMode.OFF).notifyOwed()
        Mockito.verifyNoInteractions(cases, decisionLog, publisher)
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        notifier(RelevanceMode.RECORD_ONLY).notifyOwed()
        Mockito.verifyNoInteractions(cases, decisionLog, publisher)
    }

    @Test
    fun `auto-hide without its switches runs as record-only and tells no one`() {
        // No flags switch: RelevanceSwitch yields record-only (G1).
        notifier(RelevanceMode.AUTO_HIDE, admin = AdminProperties(flagsEnabled = false)).notifyOwed()
        Mockito.verifyNoInteractions(cases, decisionLog, publisher)
    }

    @Test
    fun `each owed note gets its approved text`() {
        val n = notifier(RelevanceMode.AUTO_HIDE)
        val by = "DEC 02 · 2026" // 2026-11-18 in Manila plus 14 days
        assertEquals(RelevanceNotices.mixed("Cebu Run", 300, 1200, 45, by), n.noteFor(row("verdict", "MIXED"), by))
        assertEquals(RelevanceNotices.junkDump("Cebu Run", 300, 1200, 45, by), n.noteFor(row("verdict", "JUNK_DUMP"), by))
        assertEquals(RelevanceNotices.wrongEvent("Cebu Run", 1200, "MAR 04 · 2026", by), n.noteFor(row("verdict", "WRONG_EVENT"), by))
        assertEquals(RelevanceNotices.copied("Cebu Run", 1200, by), n.noteFor(row("verdict", "COPIED"), by))
        assertEquals(RelevanceNotices.adminHide("Cebu Run", by), n.noteFor(row("verdict", "CANT_TELL"), by))
        assertEquals(RelevanceNotices.blurry("Cebu Run", 900, 7), n.noteFor(row("blurry", "FITS"), by))
        assertEquals(RelevanceNotices.restoredAfterScan("Cebu Run"), n.noteFor(row("restored", "FITS"), by))
        assertEquals(RelevanceNotices.restoredAfterRecheck("Cebu Run"),
            n.noteFor(row("restored", "JUNK_DUMP", closeReason = "RECHECK_LIFTED"), by))
        assertEquals("DEC 02 · 2026", n.appealBy(clock.instant().atOffset(ZoneOffset.UTC)))
    }

    private fun row(note: String, verdict: String, closeReason: String? = null) = object : OwedNote {
        override val caseId = UUID.randomUUID()
        override val photographerId = UUID.randomUUID()
        override val note = note
        override val eventName = "Cebu Run"
        override val eventDate = "2026-03-04"
        override val verdict = verdict
        override val closeReason = closeReason
        override val batchPhotos = 1200
        override val measured = 300
        override val junk = 45
        override val scanMeasured = 900
        override val scanBlurry = 7
    }

    private fun notifier(mode: RelevanceMode, admin: AdminProperties = AdminProperties(flagsEnabled = true)): RelevanceNotifier {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        val switch = RelevanceSwitch(props, admin, AiApiProperties(enabled = true), rekognition)
        return RelevanceNotifier(cases, decisionLog, switch, props, clock, testTransactionTemplate(), publisher)
    }
}
