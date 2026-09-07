package com.quickpitik.service.photographer

import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoSalesProjection
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// Trash / restore / permanent-delete of a photographer's own photos. Ownership
// is enforced by findAllByIdInAndPhotographerIdAndEventId (mocked here to
// return only the owned set), so these tests focus on the state transitions,
// the photo-count bookkeeping, and the sold-photo guard on permanent delete.
class PhotographerPhotoServiceTest {

    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventPhotographerRepository: EventPhotographerRepository
    private lateinit var orderItemRepository: OrderItemRepository
    private lateinit var storageService: StorageService

    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventPhotographerRepository = Mockito.mock(EventPhotographerRepository::class.java)
        orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
    }

    private fun service() =
        PhotographerPhotoService(photoRepository, eventPhotographerRepository, orderItemRepository, storageService)

    private fun photo(
        id: UUID = UUID.randomUUID(),
        deletedAt: OffsetDateTime? = null,
    ): Photo = Photo(
        id = id,
        eventId = eventId,
        photographerId = photographerId,
        s3Key = "events/$eventId/photos/$id/original.jpg",
        pricePhp = BigDecimal("125.00"),
        status = PhotoStatus.LIVE,
    ).also {
        it.deletedAt = deletedAt
        it.watermarkS3Key = "events/$eventId/photos/$id/watermark.jpg"
        it.thumbnailS3Key = "events/$eventId/photos/$id/watermark.jpg"
    }

    private fun stubLoad(vararg photos: Photo) {
        Mockito.`when`(photoRepository.findAllByIdInAndPhotographerIdAndEventId(anyArg(), anyArg(), anyArg()))
            .thenReturn(photos.toList())
    }

    private fun stubCount(ep: EventPhotographer) {
        Mockito.`when`(eventPhotographerRepository.findById(anyArg())).thenReturn(Optional.of(ep))
    }

    private fun sales(photoId: UUID, count: Long): PhotoSalesProjection = object : PhotoSalesProjection {
        override val photoId = photoId
        override val salesCount = count
    }

    @Test
    fun `trash stamps deleted_at and drops the active photo count`() {
        val p1 = photo()
        val p2 = photo()
        stubLoad(p1, p2)
        val ep = EventPhotographer(id = EventPhotographerId(eventId, photographerId), photoCount = 5)
        stubCount(ep)

        val affected = service().trash(photographerId, eventId, listOf(p1.id, p2.id))

        assertEquals(2, affected)
        assertNotNull(p1.deletedAt)
        assertNotNull(p2.deletedAt)
        assertEquals(3, ep.photoCount)
        Mockito.verify(photoRepository).saveAll(anyArg<List<Photo>>())
    }

    @Test
    fun `trash is a no-op for photos already in the trash`() {
        val active = photo()
        val alreadyTrashed = photo(deletedAt = OffsetDateTime.now())
        stubLoad(active, alreadyTrashed)
        val ep = EventPhotographer(id = EventPhotographerId(eventId, photographerId), photoCount = 5)
        stubCount(ep)

        val affected = service().trash(photographerId, eventId, listOf(active.id, alreadyTrashed.id))

        assertEquals(1, affected)
        assertEquals(4, ep.photoCount)
    }

    @Test
    fun `restore clears deleted_at and restores the count`() {
        val trashed = photo(deletedAt = OffsetDateTime.now())
        stubLoad(trashed)
        val ep = EventPhotographer(id = EventPhotographerId(eventId, photographerId), photoCount = 3)
        stubCount(ep)

        val affected = service().restore(photographerId, eventId, listOf(trashed.id))

        assertEquals(1, affected)
        assertNull(trashed.deletedAt)
        assertEquals(4, ep.photoCount)
    }

    @Test
    fun `permanent delete removes the objects and row for an unsold trashed photo`() {
        val trashed = photo(deletedAt = OffsetDateTime.now())
        stubLoad(trashed)
        Mockito.`when`(orderItemRepository.countSalesByPhotoIds(anyArg())).thenReturn(emptyList())

        val result = service().permanentDelete(photographerId, eventId, listOf(trashed.id))

        assertEquals(listOf(trashed.id), result.deleted)
        assertEquals(emptyList(), result.blockedSold)
        // Original + watermark are distinct keys (thumbnail == watermark, deduped).
        Mockito.verify(storageService).delete(trashed.s3Key)
        Mockito.verify(storageService).delete(trashed.watermarkS3Key!!)
        Mockito.verify(photoRepository).delete(trashed)
    }

    @Test
    fun `permanent delete refuses a sold photo and never touches storage`() {
        val trashed = photo(deletedAt = OffsetDateTime.now())
        stubLoad(trashed)
        Mockito.`when`(orderItemRepository.countSalesByPhotoIds(anyArg()))
            .thenReturn(listOf(sales(trashed.id, 2)))

        val result = service().permanentDelete(photographerId, eventId, listOf(trashed.id))

        assertEquals(emptyList(), result.deleted)
        assertEquals(listOf(trashed.id), result.blockedSold)
        Mockito.verify(storageService, Mockito.never()).delete(anyArg())
        Mockito.verify(photoRepository, Mockito.never()).delete(anyArg<Photo>())
    }

    @Test
    fun `permanent delete ignores photos that are not in the trash`() {
        val active = photo() // deleted_at is null
        stubLoad(active)

        val result = service().permanentDelete(photographerId, eventId, listOf(active.id))

        assertEquals(emptyList(), result.deleted)
        assertEquals(emptyList(), result.blockedSold)
        Mockito.verifyNoInteractions(orderItemRepository, storageService)
        Mockito.verify(photoRepository, Mockito.never()).delete(anyArg<Photo>())
    }
}
