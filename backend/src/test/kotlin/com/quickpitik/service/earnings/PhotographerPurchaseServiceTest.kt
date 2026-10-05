package com.quickpitik.service.earnings

import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotographerEventService
import com.quickpitik.service.photographer.PhotographerPhotoDtoMapper
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

class PhotographerPurchaseServiceTest {
    private val seller = UUID.randomUUID()
    private val orderItems = Mockito.mock(OrderItemRepository::class.java)
    private val orders = Mockito.mock(OrderRepository::class.java)
    private val photos = Mockito.mock(PhotoRepository::class.java)
    private val events = Mockito.mock(EventRepository::class.java)
    private val transactions = Mockito.mock(TransactionRepository::class.java)
    private val disputes = Mockito.mock(DisputeRepository::class.java)
    private val users = Mockito.mock(UserRepository::class.java)
    private val eventService = Mockito.mock(PhotographerEventService::class.java)
    private val photoMapper = Mockito.mock(PhotographerPhotoDtoMapper::class.java)
    private val service = PhotographerPurchaseService(
        orderItems, orders, photos, events, transactions, disputes, users, eventService, photoMapper,
    )

    @Test
    fun `gallery keeps total when requested offset is beyond final page`() {
        val eventId = UUID.randomUUID()
        val params = PaginationParams(100, 20)
        Mockito.`when`(orderItems.pagePurchasedPhotosForPhotographer(anyArg(), anyArg(), anyArg()))
            .thenReturn(PageImpl(emptyList(), org.springframework.data.domain.PageRequest.of(5, 20), 4))

        val result = service.listPurchasedPhotos(seller, eventId, params)

        assertEquals(4L, result.total)
        assertEquals(100, result.offset)
        assertEquals(emptyList(), result.items)
        Mockito.verify(eventService).getEventDetail(seller, eventId, true)
    }

    @Test
    fun `paid purchase without sale ledger leaves kept money unknown`() {
        val event = Event(
            slug = "test-${UUID.randomUUID()}", name = "Race", date = LocalDate.of(2026, 9, 1),
            location = "Cebu", status = EventStatus.ACTIVE, createdBy = seller,
        )
        val photo = Photo(eventId = event.id, photographerId = seller, s3Key = "photo.jpg", pricePhp = BigDecimal("999.00"))
        val order = Order(
            eventId = event.id, recipientEmail = "guest.runner@example.com", paymentMethodWire = "gcash",
            totalPhp = BigDecimal("80.00"), status = OrderStatus.FULFILLED,
            paidAt = OffsetDateTime.parse("2026-09-01T10:00:00Z"),
        )
        val item = OrderItem(OrderItemId(order.id, photo.id), BigDecimal("100.00"), BigDecimal("20.00"))
        Mockito.`when`(photos.findFirstByIdAndPhotographerId(photo.id, seller)).thenReturn(photo)
        Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))
        Mockito.`when`(orderItems.pagePurchasesForPhotographerPhoto(anyArg(), anyArg(), anyArg()))
            .thenReturn(PageImpl(listOf(item)))
        Mockito.`when`(orders.findAllById(listOf(order.id))).thenReturn(listOf(order))
        Mockito.`when`(transactions.findByPhotographerIdAndPhotoIdAndOrderIdIn(seller, photo.id, listOf(order.id)))
            .thenReturn(emptyList())
        Mockito.`when`(disputes.findByOrderIdIn(listOf(order.id))).thenReturn(emptyList())
        Mockito.`when`(users.findAllById(emptySet<UUID>())).thenReturn(emptyList())
        Mockito.`when`(photoMapper.toLibraryDto(photo, 1L)).thenReturn(PhotographerLibraryPhotoDto(
            photo.id, null, "live", 1, photo.uploadedAt, 0, "default", null,
        ))

        val result = service.listPhotoPurchases(seller, photo.id, null, PaginationParams.of(null, null))

        assertEquals(1L, result.total)
        assertEquals(BigDecimal("80.00"), result.items.single().paidPhp)
        assertEquals("Guest R.", result.items.single().buyer)
        assertNull(result.items.single().netKeptPhp)
        assertEquals(result.items.single(), result.selectedPurchase)
    }

    private fun <T> anyArg(): T = Mockito.any()
}
