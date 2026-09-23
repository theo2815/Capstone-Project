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
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
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

    @ParameterizedTest
    @EnumSource(value = OrderStatus::class, names = ["PAID", "FULFILLED"])
    fun `a settled order inside the clearing window can file a refund`(status: OrderStatus) {
        val (service, disputes, order, photoId) = paidOrder(paidDaysAgo = 6, status = status)
        Mockito.`when`(disputes.save(anyArg())).thenAnswer { it.arguments[0] }

        service.submit(order.userId!!, order.id, RefundRequest(photoIds = listOf(photoId), reason = "wrong_runner", note = ""))

        Mockito.verify(disputes).save(anyArg())
    }

    @Test
    fun `a settled legacy order can file within the createdAt window`() {
        val (service, disputes, order, photoId) = paidOrder(paidDaysAgo = null, createdDaysAgo = 1)
        Mockito.`when`(disputes.save(anyArg())).thenAnswer { it.arguments[0] }

        service.submit(order.userId!!, order.id, RefundRequest(listOf(photoId), "wrong_runner", ""))

        Mockito.verify(disputes).save(anyArg())
    }

    @ParameterizedTest
    @EnumSource(value = OrderStatus::class, names = ["PENDING", "EXPIRED", "REFUNDED"])
    fun `an unsettled order cannot file a refund even with paidAt`(status: OrderStatus) {
        val (service, disputes, order, photoId, publisher) = paidOrder(paidDaysAgo = 1, status = status)
        Mockito.`when`(disputes.save(anyArg())).thenAnswer { it.arguments[0] }

        val error = assertFailsWith<ConflictException> {
            service.submit(order.userId!!, order.id, RefundRequest(listOf(photoId), "wrong_runner", ""))
        }

        assertEquals(ErrorCodes.CONFLICT, error.code)
        Mockito.verify(disputes, Mockito.never()).save(anyArg())
        Mockito.verifyNoInteractions(publisher)
    }

    @Test
    fun `a foreign order stays hidden before eligibility is checked`() {
        val (service, disputes, order, photoId, publisher) = paidOrder(paidDaysAgo = 1, status = OrderStatus.PENDING)

        val error = assertFailsWith<NotFoundException> {
            service.submit(UUID.randomUUID(), order.id, RefundRequest(listOf(photoId), "wrong_runner", ""))
        }

        assertEquals(ErrorCodes.ORDER_NOT_FOUND, error.code)
        Mockito.verifyNoInteractions(disputes, publisher)
    }

    private data class Fixture(
        val service: RefundService,
        val disputes: DisputeRepository,
        val order: Order,
        val photoId: UUID,
        val publisher: ApplicationEventPublisher,
    )

    private fun paidOrder(paidDaysAgo: Long?, createdDaysAgo: Long = 0, status: OrderStatus = OrderStatus.FULFILLED): Fixture {
        val photoId = UUID.randomUUID()
        val order = Order(
            userId = UUID.randomUUID(),
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@example.com",
            paymentMethodWire = PaymentMethod.QRPH.wire,
            status = status,
            totalPhp = BigDecimal("125.00"),
            createdAt = OffsetDateTime.now().minusDays(createdDaysAgo),
        ).apply { paidAt = paidDaysAgo?.let { OffsetDateTime.now().minusDays(it) } }
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val disputes = Mockito.mock(DisputeRepository::class.java)
        val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        Mockito.`when`(orders.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(items.findByIdOrderId(order.id)).thenReturn(
            listOf(OrderItem(OrderItemId(order.id, photoId), BigDecimal("125.00"))),
        )
        val service = RefundService(
            orders,
            items,
            disputes,
            Mockito.mock(AdminDecisionLogRepository::class.java),
            publisher,
            PlatformProperties(),
            Mockito.mock(PhotoRepository::class.java),
            Mockito.mock(UserRepository::class.java),
            Clock.systemUTC(),
        )
        return Fixture(service, disputes, order, photoId, publisher)
    }

    private fun <T> anyArg(): T = Mockito.any()
}
