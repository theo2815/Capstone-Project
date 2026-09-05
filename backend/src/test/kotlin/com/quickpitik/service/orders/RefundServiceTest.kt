package com.quickpitik.service.orders

import com.quickpitik.dto.orders.RefundRequest
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import java.math.BigDecimal
import java.util.Optional
import java.util.UUID
import kotlin.test.assertFailsWith

// Free checkout (2026-09-05): a 100% giveaway charged nothing, so a refund
// request on it is refused before a dispute row exists — the admin path
// would otherwise reach PayMongo with no payment to refund against.
class RefundServiceTest {
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
        )

        assertFailsWith<ConflictException> {
            service.submit(runnerId, order.id, RefundRequest(photoIds = listOf(photoId), reason = "wrong_runner", note = ""))
        }
        Mockito.verify(disputes, Mockito.never()).save(anyArg())
    }

    private fun <T> anyArg(): T = Mockito.any()
}
