package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceAppealRepository
import com.quickpitik.repository.OwnCaseRow
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.ai.RekognitionAiClient
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.context.ApplicationEventPublisher
import java.time.Clock
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// RV-3.2a, RVF-14 (Q3): the photographer's routes work in every mode for a case applied in auto-hide; a case applied in
// record-only has no page and no appeal in any mode, and nothing is written for it.
class RelevanceAppealServiceTest {
    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val appeals = Mockito.mock(RelevanceAppealRepository::class.java)
    private val alerts = Mockito.mock(RelevanceAlertRepository::class.java)
    private val notifier = Mockito.mock(RelevanceNotifier::class.java)
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
    private val confirmer = Mockito.mock(RelevanceConfirmer::class.java)
    private val decisionLog = Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java)

    @Test
    fun `with the mode off a case applied in record-only has no page and no appeal`() {
        assertNothingRuns(RelevanceMode.OFF)
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        assertNothingRuns(RelevanceMode.RECORD_ONLY)
    }

    private fun assertNothingRuns(mode: RelevanceMode) {
        val service = service(mode)
        val id = UUID.randomUUID()
        val photographer = UUID.randomUUID()
        val row = Mockito.mock(OwnCaseRow::class.java)
        Mockito.`when`(row.appliedMode).thenReturn("RECORD_ONLY")
        Mockito.`when`(row.stage).thenReturn("APPLIED")
        Mockito.`when`(cases.ownCase(id, photographer)).thenReturn(row)
        Mockito.`when`(cases.lockCase(id)).thenReturn(id)

        assertFailsWith<NotFoundException> { service.caseFor(photographer, id) }
        assertEquals("NOTHING_TO_APPEAL", assertFailsWith<ConflictException> { service.file(photographer, id, "mine") }.code)
        Mockito.verify(appeals, Mockito.never()).saveAndFlush(Mockito.any())
        Mockito.verifyNoInteractions(alerts, notifier, publisher, confirmer, decisionLog)
    }

    private fun service(mode: RelevanceMode): RelevanceAppealService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        @Suppress("UNCHECKED_CAST")
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), provider)
        return RelevanceAppealService(cases, appeals, alerts, notifier, switch, props, Clock.systemUTC(), publisher, confirmer, decisionLog,
            Mockito.mock(com.quickpitik.service.relevance.RelevanceHoldService::class.java),
            Mockito.mock(org.springframework.transaction.support.TransactionTemplate::class.java), Mockito.mock(AdminRelevanceService::class.java))
    }
}
