package com.quickpitik.service.events

import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.repository.EventPhotoAlertRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID

// Mirrors PhotoIndexingTriggerTest: the sweep is gated on ai-api being enabled,
// no-ops on an empty backlog, and drives the notifier once per pending opt-in
// across the upload window plus its final indexing-grace day.
class EventPhotosReadySweepTest {

    private lateinit var alertRepository: EventPhotoAlertRepository
    private lateinit var notifier: EventPhotosReadyNotifier

    @BeforeEach
    fun setUp() {
        alertRepository = Mockito.mock(EventPhotoAlertRepository::class.java)
        notifier = Mockito.mock(EventPhotosReadyNotifier::class.java)
    }

    private fun sweep(props: AiApiProperties) = EventPhotosReadySweep(alertRepository, notifier, props)

    private fun alert() = EventPhotoAlert(eventId = UUID.randomUUID(), userId = UUID.randomUUID())

    @Test
    fun `sweep when ai-api disabled does nothing`() {
        sweep(AiApiProperties(enabled = false)).sweep()

        Mockito.verifyNoInteractions(alertRepository)
        Mockito.verifyNoInteractions(notifier)
    }

    @Test
    fun `sweep with no pending opt-ins does nothing`() {
        Mockito.`when`(alertRepository.findPendingInWindow(anyArg(), anyArg(), anyArg()))
            .thenReturn(emptyList())

        sweep(AiApiProperties(enabled = true)).sweep()

        Mockito.verifyNoInteractions(notifier)
    }

    @Test
    fun `sweep notifies each pending opt-in and windows back three days`() {
        val a1 = alert()
        val a2 = alert()
        Mockito.`when`(alertRepository.findPendingInWindow(anyArg(), anyArg(), anyArg()))
            .thenReturn(listOf(a1, a2))

        // Computed the same way the sweep computes it; a midnight boundary flake
        // is a ~1-in-86400 risk and acceptable for a unit test.
        val today = LocalDate.now(ZoneId.of("Asia/Manila"))
        sweep(AiApiProperties(enabled = true)).sweep()

        Mockito.verify(notifier).notifyIfMatched(a1.id)
        Mockito.verify(notifier).notifyIfMatched(a2.id)
        // Date+4 gets one final day for accepted uploads to finish indexing.
        Mockito.verify(alertRepository)
            .findPendingInWindow(eqArg(today), eqArg(today.minusDays(4)), anyArg())
    }

    // Mockito.eq returns a platform type; wrapping keeps Kotlin's null-check off a
    // non-null parameter. Same shape as OrderReceiptEmailClaimTest.
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value

    private fun <T> anyArg(): T = Mockito.any()
}
