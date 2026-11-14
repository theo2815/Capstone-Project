package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID

class RelevanceWordWriterTest {
    private val repository = Mockito.mock(BatchRelevanceRepository::class.java)
    private val clock = Clock.fixed(Instant.parse("2003-03-04T05:06:07Z"), ZoneOffset.UTC)
    private val photoId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()
    private val batch = BatchRelevance(eventId = eventId, photographerId = photographerId)

    private fun writer(mode: RelevanceMode) = RelevanceWordWriter(repository,
        RelevanceSwitch(RelevanceProperties(mode = mode), AdminProperties(), AiApiProperties()), testTransactionTemplate(), clock)

    private fun <T> anyArg(): T = Mockito.any()

    @Test fun `nothing is written while the mode is off`() {
        writer(RelevanceMode.OFF).record(photoId, eventId, photographerId, listOf("CEBU", "MARATHON"))
        Mockito.verifyNoInteractions(repository)
    }

    @Test fun `a photo already counted adds nothing`() {
        Mockito.`when`(repository.findByEventIdAndPhotographerId(eventId, photographerId)).thenReturn(batch)
        Mockito.`when`(repository.markWordsCounted(anyArg(), anyArg(), anyArg())).thenReturn(1, 0)
        val w = writer(RelevanceMode.RECORD_ONLY)

        w.record(photoId, eventId, photographerId, listOf("marathon", "Cebu", "1234"))
        Mockito.verify(repository).ensureBatch(eventId, photographerId)
        Mockito.verify(repository).addWords(batch.id, arrayOf("CEBU", "MARATHON"))

        w.record(photoId, eventId, photographerId, listOf("marathon", "Cebu"))
        Mockito.verify(repository, Mockito.times(2)).markWordsCounted(anyArg(), anyArg(), anyArg())
        Mockito.verify(repository, Mockito.times(1)).addWords(anyArg(), anyArg())
    }
}
