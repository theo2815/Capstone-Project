package com.quickpitik.service.cart

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.CartItemEntity
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.CartItemRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Runner-flow audit (2026-05-27), "Cart + saved events" items 6/7.
//
// merge() re-homed a guest cart without checking either the photo status or the
// event status, so PROCESSING/HIDDEN photos and ARCHIVED-event photos landed in
// the authed cart on login — both of which add() already blocks (add() blocked
// ARCHIVED; the photo-status check was missing on BOTH paths).
//
// merge skips where add throws — see the docblock on CartService.merge.
class CartServiceTest {

    private lateinit var cartItemRepository: CartItemRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var storageService: StorageService
    private lateinit var service: CartService

    private val userId = UUID.randomUUID()
    private val written = mutableListOf<CartItemEntity>()

    @BeforeEach
    fun setUp() {
        cartItemRepository = Mockito.mock(CartItemRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        service = CartService(
            cartItemRepository,
            photoRepository,
            eventRepository,
            storageService,
            StorageProperties(),
        )

        written.clear()
        Mockito.`when`(cartItemRepository.save(anyArg<CartItemEntity>())).thenAnswer {
            val row = it.arguments[0] as CartItemEntity
            written += row
            row
        }
        // Empty on both reads: merge() builds `existing` from the first call and
        // list() hydrates from the second. Empty keeps these tests on what merge
        // wrote rather than on presigned-URL hydration.
        Mockito.`when`(cartItemRepository.findByUserId(userId)).thenReturn(emptyList())
    }

    @Test
    fun `merge skips a non-LIVE photo`() {
        val event = event(EventStatus.ACTIVE)
        val hidden = photo(event.id, status = PhotoStatus.HIDDEN)
        stub(listOf(hidden), listOf(event))

        service.merge(userId, listOf(hidden.id to event.id))

        assertEquals(emptyList(), written)
    }

    @Test
    fun `merge skips a photo from an ARCHIVED event`() {
        val archived = event(EventStatus.ARCHIVED)
        val live = photo(archived.id)
        stub(listOf(live), listOf(archived))

        service.merge(userId, listOf(live.id to archived.id))

        assertEquals(emptyList(), written)
    }

    @Test
    fun `merge still writes the good rows in a mixed batch`() {
        val open = event(EventStatus.ACTIVE)
        val archived = event(EventStatus.ARCHIVED)
        val good = photo(open.id)
        val hidden = photo(open.id, status = PhotoStatus.HIDDEN)
        val fromArchived = photo(archived.id)
        stub(listOf(good, hidden, fromArchived), listOf(open, archived))

        service.merge(
            userId,
            listOf(good.id to open.id, hidden.id to open.id, fromArchived.id to archived.id),
        )

        assertEquals(listOf(good.id), written.map { it.id.photoId })
    }

    @Test
    fun `merge snapshots the current photo price`() {
        val open = event(EventStatus.ACTIVE)
        val good = photo(open.id, price = BigDecimal("150.00"))
        stub(listOf(good), listOf(open))

        service.merge(userId, listOf(good.id to open.id))

        assertEquals(BigDecimal("150.00"), written.single().pricePhpAtAdd)
    }

    @Test
    fun `add rejects a non-LIVE photo with 404`() {
        val event = event(EventStatus.ACTIVE)
        val hidden = photo(event.id, status = PhotoStatus.HIDDEN)
        Mockito.`when`(photoRepository.findById(hidden.id)).thenReturn(Optional.of(hidden))

        val ex = assertFailsWith<NotFoundException> { service.add(userId, hidden.id, event.id) }

        assertEquals(ErrorCodes.PHOTO_NOT_FOUND, ex.code)
        assertEquals(emptyList(), written)
    }

    private fun stub(photos: List<Photo>, events: List<Event>) {
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(photos)
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(events)
    }

    private fun event(status: EventStatus): Event = Event(
        slug = "cebu-marathon-${UUID.randomUUID()}",
        name = "Cebu Marathon",
        date = LocalDate.of(2026, 1, 11),
        location = "Cebu City, Cebu",
        status = status,
    )

    private fun photo(
        eventId: UUID,
        status: PhotoStatus = PhotoStatus.LIVE,
        price: BigDecimal = BigDecimal("125.00"),
    ): Photo = Photo(
        eventId = eventId,
        s3Key = "photos/${UUID.randomUUID()}.jpg",
        status = status,
        pricePhp = price,
    )

    private fun <T> anyArg(): T = Mockito.any()
}
