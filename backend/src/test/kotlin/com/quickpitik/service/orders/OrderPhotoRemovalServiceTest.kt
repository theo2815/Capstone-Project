package com.quickpitik.service.orders

import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.removal.PhotoRemovalHistoryRecorder
import com.quickpitik.service.runner.RunnerMessagesService
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import java.math.BigDecimal
import java.time.Instant
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID

// Unit boundaries supplement the PostgreSQL notice RED and transaction checks.
class OrderPhotoRemovalServiceTest {
    private val photos = mock(PhotoRepository::class.java)
    private val orders = mock(OrderRepository::class.java)
    private val lines = mock(OrderItemRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val history = mock(PhotoRemovalHistoryRepository::class.java)
    private val recorder = mock(PhotoRemovalHistoryRecorder::class.java)
    private val inbox = mock(RunnerMessagesService::class.java)
    private val service = OrderPhotoRemovalService(photos, orders, lines, items, history, recorder, inbox)
    private val photoId = UUID.randomUUID()
    private val requestId = UUID.randomUUID()
    private val now = Instant.parse("2026-10-07T00:00:00Z")
    private fun <T> arg(): T = any()
    private fun item() = PhotoRemovalItem(requestId = requestId, photoIdSnapshot = photoId,
        state = RemovalItemState.REMOVAL_AUTHORIZED, matchSource = RemovalMatchSource.MANUAL)
    private fun order(paid: Boolean = true, guest: Boolean = false) = Order(eventId = UUID.randomUUID(),
        userId = if (guest) null else UUID.randomUUID(), recipientEmail = "buyer@test.local", paymentMethodWire = "gcash",
        totalPhp = BigDecimal("125.00"), paidAt = if (paid) OffsetDateTime.now() else null)

    @Test fun `review never creates buyer removal work`() {
        service.onRemovalAuthorized(photoId, requestId, now)
        verifyNoInteractions(orders, lines, items, history, recorder, inbox)
    }

    @Test fun `authorization notifies paid buyers while pending orders wait for settlement`() {
        val paid = order()
        val pending = order(paid = false)
        `when`(photos.isRemoved(photoId)).thenReturn(true)
        `when`(items.findByRequestIdOrderByIdAsc(requestId)).thenReturn(listOf(item()))
        `when`(orders.findReferencingPhoto(photoId)).thenReturn(listOf(paid, pending))
        service.onRemovalAuthorized(photoId, requestId, now)
        verify(inbox).pushRemovalNotice(paid.userId!!, paid.id, photoId, requestId, now)
        verifyNoMoreInteractions(inbox)
        verify(orders, never()).save(arg<Order>())
        verifyNoInteractions(lines, recorder)
    }

    @Test fun `guest replay keeps the existing durable task`() {
        val paid = order(guest = true)
        `when`(photos.isRemoved(photoId)).thenReturn(true)
        `when`(items.findByRequestIdOrderByIdAsc(requestId)).thenReturn(listOf(item()))
        `when`(orders.findReferencingPhoto(photoId)).thenReturn(listOf(paid))
        `when`(history.hasGuestBuyerNotice(photoId, paid.id.toString())).thenReturn(true)
        service.onRemovalAuthorized(photoId, requestId, now)
        verifyNoInteractions(recorder, inbox)
    }

    @Test fun `settlement routes only removed lines to the original removal case`() {
        val paid = order()
        val retained = UUID.randomUUID()
        val orderLines = listOf(OrderItem(OrderItemId(paid.id, photoId), BigDecimal.ONE),
            OrderItem(OrderItemId(paid.id, retained), BigDecimal.ONE))
        `when`(lines.findByIdOrderIdIn(listOf(paid.id))).thenReturn(orderLines)
        `when`(lines.findByIdOrderId(paid.id)).thenReturn(orderLines)
        `when`(orders.findById(paid.id)).thenReturn(Optional.of(paid))
        `when`(photos.isRemoved(photoId)).thenReturn(true)
        `when`(items.findByPhotoIdSnapshotAndStateInOrderByAdminDecidedAtAscIdAsc(photoId,
            setOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED))).thenReturn(listOf(item()))
        service.onOrderSettled(paid.id, now)
        verify(inbox).pushRemovalNotice(paid.userId!!, paid.id, photoId, requestId, now)
        verifyNoMoreInteractions(inbox)
        verify(orders, never()).save(arg<Order>())
        verify(lines, never()).save(arg<OrderItem>())
    }

    @Test fun `settlement without paidAt produces no notice`() {
        val pending = order(paid = false)
        `when`(orders.findById(pending.id)).thenReturn(Optional.of(pending))
        service.onOrderSettled(pending.id, now)
        verifyNoInteractions(history, recorder, inbox)
    }

    @Test fun `sibling photo locks are unique and ordered across orders`() {
        val first = UUID(0, 1)
        val second = UUID(0, 2)
        val ids = listOf(UUID.randomUUID(), UUID.randomUUID())
        `when`(lines.findByIdOrderIdIn(ids)).thenReturn(listOf(
            OrderItem(OrderItemId(ids[0], second), BigDecimal.ONE),
            OrderItem(OrderItemId(ids[0], first), BigDecimal.ONE),
            OrderItem(OrderItemId(ids[1], second), BigDecimal.ONE)))
        service.lockSettlementPhotos(ids)
        verify(photos).findAllByIdForUpdate(listOf(first, second))
        verifyNoInteractions(orders, items, history, recorder, inbox)
    }
}
