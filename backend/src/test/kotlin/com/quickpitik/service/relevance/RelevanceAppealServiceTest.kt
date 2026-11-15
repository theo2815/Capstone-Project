package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceAppealRepository
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

// RV-3.2a: with the mode off, or in record-only, the appeal routes run no query and change nothing.
class RelevanceAppealServiceTest {
    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val appeals = Mockito.mock(RelevanceAppealRepository::class.java)
    private val alerts = Mockito.mock(RelevanceAlertRepository::class.java)
    private val notifier = Mockito.mock(RelevanceNotifier::class.java)
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
    private val confirmer = Mockito.mock(RelevanceConfirmer::class.java)
    private val decisionLog = Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java)

    @Test
    fun `with the mode off nothing runs`() {
        assertNothingRuns(RelevanceMode.OFF)
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        assertNothingRuns(RelevanceMode.RECORD_ONLY)
    }

    private fun assertNothingRuns(mode: RelevanceMode) {
        val service = service(mode)
        val id = UUID.randomUUID()
        assertFailsWith<NotFoundException> { service.caseFor(UUID.randomUUID(), id) }
        assertEquals("NOTHING_TO_APPEAL", assertFailsWith<ConflictException> { service.file(UUID.randomUUID(), id, "mine") }.code)
        // RV-3.2 part b: the admin's evidence look and decision change nothing either.
        assertEquals("RELEVANCE_ACTIONS_OFF", assertFailsWith<ApiException> { service.evidenceOpened(UUID.randomUUID(), id) }.code)
        assertEquals("RELEVANCE_ACTIONS_OFF", assertFailsWith<ApiException> { service.decide(UUID.randomUUID(), id, "LOST", null) }.code)
        Mockito.verifyNoInteractions(cases, appeals, alerts, notifier, publisher, confirmer, decisionLog,
            Mockito.mock(com.quickpitik.service.relevance.RelevanceHoldService::class.java))
    }

    private fun service(mode: RelevanceMode): RelevanceAppealService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        @Suppress("UNCHECKED_CAST")
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), provider)
        return RelevanceAppealService(cases, appeals, alerts, notifier, switch, props, Clock.systemUTC(), publisher, confirmer, decisionLog,
            Mockito.mock(com.quickpitik.service.relevance.RelevanceHoldService::class.java))
    }
}
