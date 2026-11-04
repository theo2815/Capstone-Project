package com.quickpitik.service.events

import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.repository.EventPhotoAlertRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageRequest
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertFailsWith

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

    private fun sweep(props: AiApiProperties) = EventPhotosReadySweep(alertRepository, notifier, props, FIXED_CLOCK)

    private fun alert() = EventPhotoAlert(eventId = UUID.randomUUID(), userId = UUID.randomUUID())

    @Test
    fun `sweep when ai-api disabled does nothing`() {
        sweep(AiApiProperties(enabled = false)).sweep()

        Mockito.verifyNoInteractions(alertRepository)
        Mockito.verifyNoInteractions(notifier)
    }

    @Test
    fun `sweep with no pending opt-ins does nothing`() {
        Mockito.`when`(alertRepository.findPendingInWindow(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(emptyList())

        sweep(AiApiProperties(enabled = true)).sweep()

        Mockito.verifyNoInteractions(notifier)
    }

    @Test
    fun `sweep notifies each pending opt-in and windows back three days`() {
        val a1 = alert()
        val a2 = alert()
        Mockito.`when`(alertRepository.findPendingInWindow(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(listOf(a1, a2))

        sweep(AiApiProperties(enabled = true)).sweep()

        Mockito.verify(notifier).notifyIfMatched(a1.id)
        Mockito.verify(notifier).notifyIfMatched(a2.id)
        // Date+4 gets one final day for accepted uploads to finish indexing.
        Mockito.verify(alertRepository)
            .findPendingInWindow(
                eqArg(TODAY),
                eqArg(TODAY.minusDays(4)),
                eqArg(NOW.minus(Duration.ofMinutes(10))),
                eqArg(PageRequest.of(0, 100)),
            )
    }

    @Test
    fun `photos-ready claim ttl must be positive`() {
        assertFailsWith<IllegalArgumentException> {
            AiApiProperties(photosReadyClaimTtl = Duration.ZERO)
        }
    }

    // Mockito.eq returns a platform type; wrapping keeps Kotlin's null-check off a
    // non-null parameter. Same shape as OrderReceiptEmailClaimTest.
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value

    private fun <T> anyArg(): T = Mockito.any()

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2026-09-22T04:05:06.123456Z")
        val FIXED_CLOCK: Clock = Clock.fixed(Instant.parse("2026-09-22T04:05:06.123456789Z"), ZoneOffset.UTC)
        val TODAY: LocalDate = LocalDate.now(FIXED_CLOCK.withZone(ZoneId.of("Asia/Manila")))
    }
}
