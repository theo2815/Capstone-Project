package com.quickpitik.service.photographer

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoUploadIntentRepository
import com.quickpitik.service.orders.PaymongoCheckoutReconciler
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.support.TransactionTemplate
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PhotographerPhotoServiceTest {
    private val photos = Mockito.mock(PhotoRepository::class.java)
    private val events = Mockito.mock(EventRepository::class.java)
    private val service = PhotographerPhotoService(
        photos,
        Mockito.mock(PhotoUploadIntentRepository::class.java),
        Mockito.mock(EventPhotographerRepository::class.java),
        Mockito.mock(OrderItemRepository::class.java),
        Mockito.mock(StorageService::class.java),
        events,
        Mockito.mock(DownloadGrantRepository::class.java),
        Mockito.mock(PaymongoCheckoutReconciler::class.java),
        Mockito.mock(TransactionTemplate::class.java),
    )
    private val seller = UUID.randomUUID()
    private val photoId = UUID.randomUUID()

    @Test
    fun `removed event rejects every photo mutation`() {
        val event = event(OffsetDateTime.parse("2026-09-04T21:36:27Z"))
        Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))

        assertFailsWith<NotFoundException> { service.trash(seller, event.id, listOf(photoId)) }
        assertFailsWith<NotFoundException> { service.restore(seller, event.id, listOf(photoId)) }
        assertFailsWith<NotFoundException> { service.permanentDelete(seller, event.id, listOf(photoId)) }
    }

    @Test
    fun `available event still permits photographer photo management`() {
        val event = event(null)
        Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))

        assertEquals(0, service.trash(seller, event.id, listOf(photoId)))
        assertEquals(0, service.restore(seller, event.id, listOf(photoId)))
        assertEquals(emptyList(), service.permanentDelete(seller, event.id, listOf(photoId)).deleted)
    }

    @Test
    fun `unknown event remains a no-op for photo mutations`() {
        val eventId = UUID.randomUUID()
        Mockito.`when`(events.findById(eventId)).thenReturn(Optional.empty())

        assertEquals(0, service.trash(seller, eventId, listOf(photoId)))
        assertEquals(0, service.restore(seller, eventId, listOf(photoId)))
        assertEquals(emptyList(), service.permanentDelete(seller, eventId, listOf(photoId)).deleted)
    }

    private fun event(deletedAt: OffsetDateTime?) = Event(
        slug = "photo-management-${UUID.randomUUID()}",
        name = "Photo management",
        date = LocalDate.of(2026, 9, 1),
        location = "Cebu",
        status = EventStatus.ACTIVE,
        deletedAt = deletedAt,
    )
}
