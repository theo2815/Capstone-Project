package com.quickpitik.service.orders

import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.orders.RefundRequest
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import java.math.BigDecimal
import java.time.Clock
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertFailsWith
import kotlin.test.assertEquals

// Free checkout (2026-09-05): a 100% giveaway charged nothing, so a refund
// request on it is refused before a dispute row exists — the admin path
// would otherwise reach PayMongo with no payment to refund against.
class RefundServiceTest {
    @Test
    fun `a pending provider refund cannot be withdrawn or reopened`() {
        val runnerId = UUID.randomUUID()
        val dispute = Dispute(
            orderId = UUID.randomUUID(),
            photoId = UUID.randomUUID(),
            runnerId = runnerId,
            reasonWire = "other",
        ).apply {
            refundStatus = "processing"
            refundAmountPhp = BigDecimal("125.00")
        }
        val disputes = Mockito.mock(DisputeRepository::class.java)
        Mockito.`when`(disputes.findById(dispute.id)).thenReturn(Optional.of(dispute))
        Mockito.`when`(disputes.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        val service = RefundService(
            Mockito.mock(OrderRepository::class.java),
            Mockito.mock(OrderItemRepository::class.java),
            disputes,
            Mockito.mock(AdminDecisionLogRepository::class.java),
            Mockito.mock(ApplicationEventPublisher::class.java),
            PlatformProperties(),
            Mockito.mock(PhotoRepository::class.java),
            Mockito.mock(UserRepository::class.java),
            Clock.systemUTC(),
        )

        val error = assertFailsWith<ApiException> { service.withdraw(runnerId, dispute.id) }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, error.code)
        assertEquals(DisputeStatus.OPEN, dispute.status)
        assertEquals(BigDecimal("125.00"), dispute.refundAmountPhp)
        Mockito.verify(disputes).findByIdForUpdate(dispute.id)
        Mockito.verify(disputes, Mockito.never()).save(dispute)
    }

    @Test
    fun `a photo that was charged nothing cannot be disputed`() {
        val runnerId = UUID.randomUUID()
        val photoId = UUID.randomUUID()
        val order = Order(
            userId = runnerId,
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@example.com",
            paymentMethodWire = PaymentMethod.QRPH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal.ZERO,
        )
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val disputes = Mockito.mock(DisputeRepository::class.java)
        Mockito.`when`(orders.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(items.findByIdOrderId(order.id)).thenReturn(
            listOf(OrderItem(OrderItemId(order.id, photoId), BigDecimal("125.00"), discountPhp = BigDecimal("125.00"))),
        )
        val service = RefundService(
            orders,
            items,
            disputes,
            Mockito.mock(AdminDecisionLogRepository::class.java),
            Mockito.mock(ApplicationEventPublisher::class.java),
            PlatformProperties(),
            Mockito.mock(PhotoRepository::class.java),
            Mockito.mock(UserRepository::class.java),
            Clock.systemUTC(),
        )

        assertFailsWith<ConflictException> {
            service.submit(runnerId, order.id, RefundRequest(photoIds = listOf(photoId), reason = "wrong_runner", note = ""))
        }
        Mockito.verify(disputes, Mockito.never()).save(anyArg())
    }

    // M-06: disputes close with the payout clearing window, so a refund can
    // never land after the sale has become withdrawable.
    @Test
    fun `a refund request after the clearing window is refused`() {
        val (service, disputes, order, photoId) = paidOrder(paidDaysAgo = 8)

        assertFailsWith<ConflictException> {
            service.submit(order.userId!!, order.id, RefundRequest(photoIds = listOf(photoId), reason = "wrong_runner", note = ""))
        }
        Mockito.verify(disputes, Mockito.never()).save(anyArg())
    }

    // Minting clears such a sale from createdAt, so the window starts there too.
    @Test
    fun `an order with no paidAt closes its refund window from createdAt`() {
        val (service, disputes, order, photoId) = paidOrder(paidDaysAgo = null, createdDaysAgo = 8)

        assertFailsWith<ConflictException> {
            service.submit(order.userId!!, order.id, RefundRequest(photoIds = listOf(photoId), reason = "wrong_runner", note = ""))
        }
        Mockito.verify(disputes, Mockito.never()).save(anyArg())
    }

    @Test
    fun `a refund request inside the clearing window is filed`() {
        val (service, disputes, order, photoId) = paidOrder(paidDaysAgo = 6)
        Mockito.`when`(disputes.save(anyArg())).thenAnswer { it.arguments[0] }

        service.submit(order.userId!!, order.id, RefundRequest(photoIds = listOf(photoId), reason = "wrong_runner", note = ""))

        Mockito.verify(disputes).save(anyArg())
    }

    private data class Fixture(val service: RefundService, val disputes: DisputeRepository, val order: Order, val photoId: UUID)

    private fun paidOrder(paidDaysAgo: Long?, createdDaysAgo: Long = 0): Fixture {
        val photoId = UUID.randomUUID()
        val order = Order(
            userId = UUID.randomUUID(),
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@example.com",
            paymentMethodWire = PaymentMethod.QRPH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("125.00"),
            createdAt = OffsetDateTime.now().minusDays(createdDaysAgo),
        ).apply { paidAt = paidDaysAgo?.let { OffsetDateTime.now().minusDays(it) } }
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val disputes = Mockito.mock(DisputeRepository::class.java)
        Mockito.`when`(orders.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(items.findByIdOrderId(order.id)).thenReturn(
            listOf(OrderItem(OrderItemId(order.id, photoId), BigDecimal("125.00"))),
        )
        val service = RefundService(
            orders,
            items,
            disputes,
            Mockito.mock(AdminDecisionLogRepository::class.java),
            Mockito.mock(ApplicationEventPublisher::class.java),
            PlatformProperties(),
            Mockito.mock(PhotoRepository::class.java),
            Mockito.mock(UserRepository::class.java),
            Clock.systemUTC(),
        )
        return Fixture(service, disputes, order, photoId)
    }

    private fun <T> anyArg(): T = Mockito.any()
}
