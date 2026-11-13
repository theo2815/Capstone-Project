package com.quickpitik.service.photos

import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoRemovalItemRepository
import com.quickpitik.entity.*
import com.quickpitik.exception.ConflictException
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import com.quickpitik.websocket.PhotosHiddenEvent
import java.util.UUID
import java.util.Optional
import java.math.BigDecimal
import kotlin.test.assertFailsWith

class PhotoAvailabilityServiceTest {
    @Test fun `authorization rejects a different decision actor before removal`() {
        val photos = Mockito.mock(PhotoRepository::class.java)
        val items = Mockito.mock(PhotoRemovalItemRepository::class.java)
        val photo = Photo(eventId = UUID.randomUUID(), s3Key = "synthetic/photo.jpg", pricePhp = BigDecimal.TEN)
        val item = PhotoRemovalItem(requestId = UUID.randomUUID(), photoIdSnapshot = photo.id,
            photoId = photo.id, state = RemovalItemState.REMOVAL_AUTHORIZED, matchSource = RemovalMatchSource.MANUAL,
            adminDecision = RemovalDecision.REMOVE, adminDecisionActorId = UUID.randomUUID())
        Mockito.`when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        Mockito.`when`(items.findById(item.id)).thenReturn(Optional.of(item))
        val service = PhotoAvailabilityService(photos, items, Mockito.mock(ApplicationEventPublisher::class.java))
        assertFailsWith<ConflictException> { service.authorizeRemoval(photo.id, item.id, UUID.randomUUID()) }
    }

    @Test fun `refresh locks all affected photos in UUID order before recalculating availability`() {
        val photos = Mockito.mock(PhotoRepository::class.java)
        val service = PhotoAvailabilityService(photos, Mockito.mock(PhotoRemovalItemRepository::class.java), Mockito.mock(ApplicationEventPublisher::class.java))
        val first = UUID.fromString("00000000-0000-0000-0000-000000000001")
        val second = UUID.fromString("00000000-0000-0000-0000-000000000002")
        service.refreshRemovalState(linkedSetOf(second, first))
        Mockito.verify(photos).findAllByIdForUpdate(listOf(first, second))
    }

    @Test fun `refreshRemovalState publishes only photos now in REVIEW`() {
        val photos = Mockito.mock(PhotoRepository::class.java)
        val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        val service = PhotoAvailabilityService(photos, Mockito.mock(PhotoRemovalItemRepository::class.java), publisher)
        val eventA = UUID.randomUUID()
        val eventB = UUID.randomUUID()
        val a1 = Photo(eventId = eventA, s3Key = "a1.jpg", pricePhp = BigDecimal.TEN)
        val a2 = Photo(eventId = eventA, s3Key = "a2.jpg", pricePhp = BigDecimal.TEN)
        val b1 = Photo(eventId = eventB, s3Key = "b1.jpg", pricePhp = BigDecimal.TEN)
        val restored = Photo(eventId = eventA, s3Key = "restored.jpg", pricePhp = BigDecimal.TEN)
        val ids = listOf(a1.id, a2.id, b1.id, restored.id).sorted()
        Mockito.`when`(photos.findAllByIdInAndRemovalState(ids, RemovalState.REVIEW)).thenReturn(listOf(a1, a2, b1))

        service.refreshRemovalState(ids.toSet())

        val published = org.mockito.ArgumentCaptor.forClass(Any::class.java)
        Mockito.verify(publisher, Mockito.times(2)).publishEvent(published.capture() ?: Any())
        val hidden = published.allValues.filterIsInstance<PhotosHiddenEvent>().associate { it.eventId to it.photoIds.toSet() }
        kotlin.test.assertEquals(mapOf(eventA to setOf(a1.id, a2.id), eventB to setOf(b1.id)), hidden)
    }

    @Test fun `refreshRemovalState publishes nothing when no photo is in REVIEW`() {
        val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        val service = PhotoAvailabilityService(Mockito.mock(PhotoRepository::class.java), Mockito.mock(PhotoRemovalItemRepository::class.java), publisher)

        service.refreshRemovalState(setOf(UUID.randomUUID()))

        Mockito.verifyNoInteractions(publisher)
    }
}
