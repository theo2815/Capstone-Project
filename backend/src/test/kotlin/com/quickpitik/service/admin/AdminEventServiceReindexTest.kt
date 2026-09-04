package com.quickpitik.service.admin

import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.Mockito
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

// Admin reindex endpoint: re-drives an event's photos through AI indexing by
// resetting them to PENDING with a fresh attempt budget (the 2026-08-25 outage
// needed manual SQL for this). Scope: FAILED/PARTIAL by default; all=true also
// requeues INDEXED + SKIPPED (provider flip / AI enabled after upload).
class AdminEventServiceReindexTest {

    private val eventRepository = Mockito.mock(EventRepository::class.java)
    private val photoRepository = Mockito.mock(PhotoRepository::class.java)
    private val service = AdminEventService(
        eventRepository,
        photoRepository,
        Mockito.mock(AdminDecisionLogService::class.java),
        Mockito.mock(EventDtoMapper::class.java),
        Mockito.mock(EventCoverService::class.java),
        Mockito.mock(FaceBibProvider::class.java),
        AiApiProperties(),
        Mockito.mock(com.quickpitik.repository.PhotographerSettingsRepository::class.java),
        Mockito.mock(com.quickpitik.repository.UserRepository::class.java),
    )

    private val adminId = UUID.randomUUID()

    private fun event(deleted: Boolean = false): Event =
        Event(
            slug = "cebu-marathon",
            name = "Cebu Marathon",
            date = LocalDate.now(),
            location = "Cebu",
            status = EventStatus.ACTIVE,
        ).also { if (deleted) it.deletedAt = OffsetDateTime.now() }

    @Test
    fun `default scope requeues FAILED and PARTIAL only`() {
        val event = event()
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))
        Mockito.`when`(
            photoRepository.requeueIndexing(event.id, listOf(IndexingStatus.FAILED, IndexingStatus.PARTIAL)),
        ).thenReturn(3)

        assertEquals(3, service.reindexPhotos(adminId, event.id, all = false))
    }

    @Test
    fun `all=true also requeues INDEXED and SKIPPED`() {
        val event = event()
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))
        Mockito.`when`(
            photoRepository.requeueIndexing(
                event.id,
                listOf(IndexingStatus.FAILED, IndexingStatus.PARTIAL, IndexingStatus.INDEXED, IndexingStatus.SKIPPED),
            ),
        ).thenReturn(7)

        assertEquals(7, service.reindexPhotos(adminId, event.id, all = true))
    }

    @Test
    fun `soft-deleted or missing event - 404, nothing requeued`() {
        val deleted = event(deleted = true)
        Mockito.`when`(eventRepository.findById(deleted.id)).thenReturn(Optional.of(deleted))

        assertThrows<NotFoundException> { service.reindexPhotos(adminId, deleted.id, all = false) }
        Mockito.verifyNoInteractions(photoRepository)
    }
}
