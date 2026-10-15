package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoUploadIntentRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
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
    private val users = Mockito.mock(UserRepository::class.java)
    private val settings = Mockito.mock(PhotographerSettingsRepository::class.java)
    private val seller = UUID.randomUUID()
    private val photoId = UUID.randomUUID()
    private val service = PhotographerPhotoService(
        photos,
        Mockito.mock(EventPhotographerRepository::class.java),
        Mockito.mock(OrderItemRepository::class.java),
        events,
        Mockito.mock(DownloadGrantRepository::class.java),
        Mockito.mock(PaymongoCheckoutReconciler::class.java),
        Mockito.mock(com.quickpitik.service.photos.PhotoErasureService::class.java),
        PhotographerAccessGuard(users, settings),
    )

    init {
        Mockito.`when`(users.findById(seller)).thenReturn(
            Optional.of(User(id = seller, email = "seller@test.local", passwordHash = "x", name = "Seller", role = Role.PHOTOGRAPHER)),
        )
        Mockito.`when`(settings.findById(seller)).thenReturn(
            Optional.of(PhotographerSettings(userId = seller, verificationStatus = VerificationStatus.APPROVED)),
        )
    }

    // Security audit 2026-10-14 (D6): restoring trashed photos republishes them,
    // so a photographer who lost approval cannot do it; trash and delete stay open.
    @Test
    fun `a photographer who lost approval cannot restore photos`() {
        val event = event(null)
        Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))
        Mockito.`when`(settings.findById(seller)).thenReturn(
            Optional.of(PhotographerSettings(userId = seller, verificationStatus = VerificationStatus.INCOMPLETE)),
        )

        val ex = assertFailsWith<ApiException> { service.restore(seller, event.id, listOf(photoId)) }

        assertEquals(ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED, ex.code)
        Mockito.verify(photos, Mockito.never()).restoreOwned(Mockito.any() ?: seller, Mockito.any() ?: event.id, Mockito.anyList())
        assertEquals(0, service.trash(seller, event.id, listOf(photoId)))
    }

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

    @Test
    fun `trash locks its photos with one id-only query in id order`() {
        val event = event(null)
        Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))
        val low = UUID.fromString("10000000-0000-0000-0000-000000000001")
        val high = UUID.fromString("70000000-0000-0000-0000-000000000002")

        service.trash(seller, event.id, listOf(high, low, high))

        Mockito.verify(photos).lockIdsForUpdate(listOf(low, high))
        Mockito.verify(photos, Mockito.never()).findAllByIdForUpdate(Mockito.anyCollection())
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
