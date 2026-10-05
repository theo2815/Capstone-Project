package com.quickpitik.service.photos

import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoRemovalItemRepository
import com.quickpitik.entity.*
import com.quickpitik.exception.ConflictException
import org.junit.jupiter.api.Test
import org.mockito.Mockito
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
        val service = PhotoAvailabilityService(photos, items)
        assertFailsWith<ConflictException> { service.authorizeRemoval(photo.id, item.id, UUID.randomUUID()) }
    }

    @Test fun `refresh locks all affected photos in UUID order before recalculating availability`() {
        val photos = Mockito.mock(PhotoRepository::class.java)
        val service = PhotoAvailabilityService(photos, Mockito.mock(PhotoRemovalItemRepository::class.java))
        val first = UUID.fromString("00000000-0000-0000-0000-000000000001")
        val second = UUID.fromString("00000000-0000-0000-0000-000000000002")
        service.refreshRemovalState(linkedSetOf(second, first))
        Mockito.verify(photos).findAllByIdForUpdate(listOf(first, second))
    }
}
