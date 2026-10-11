package com.quickpitik.service.photos

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.photos.PhotoErasureOutcome
import com.quickpitik.repository.PhotoErasureJobRepository
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageRequest
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals

class PhotoErasureSweepTest {
    private val jobs = Mockito.mock(PhotoErasureJobRepository::class.java)
    private val erasure = Mockito.mock(PhotoErasureService::class.java)
    private val now = Instant.parse("2026-10-08T00:00:00Z")
    private val registry = SimpleMeterRegistry()
    private val sweep = PhotoErasureSweep(jobs, erasure, PhotoRemovalProperties(sweepBatchSize = 2),
        Clock.fixed(now, ZoneOffset.UTC), registry)

    @Test fun `sweep bounds its page and continues after one job fails`() {
        val first = UUID.randomUUID()
        val second = UUID.randomUUID()
        Mockito.`when`(jobs.findDue(now, PageRequest.of(0, 2))).thenReturn(listOf(first, second))
        Mockito.`when`(erasure.attempt(first)).thenThrow(IllegalStateException("synthetic failure"))
        assertEquals(2, sweep.sweepDue(now, 100))
        Mockito.verify(erasure).attempt(second)
    }

    @Test fun `empty limit performs no database or storage work`() {
        assertEquals(0, sweep.sweepDue(now, 0))
        Mockito.verifyNoInteractions(jobs, erasure)
    }

    @Test fun `scheduled sweep uses the injected clock and configured bound`() {
        sweep.scheduled()
        Mockito.verify(jobs).findDue(now, PageRequest.of(0, 2))
    }

    @Test fun `sweep publishes how many jobs are still open`() {
        Mockito.`when`(jobs.findDue(now, PageRequest.of(0, 2))).thenReturn(emptyList())
        Mockito.`when`(jobs.countByOutcomeNot(PhotoErasureOutcome.COMPLETE)).thenReturn(3L)
        sweep.sweepDue(now, 100)
        assertEquals(3.0, registry.get("qp.removal.erasure.open").gauge().value())
    }
}
