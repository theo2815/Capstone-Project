package com.quickpitik.service.orders

import com.quickpitik.dto.orders.PaymentWebhookRequest
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.service.earnings.TransactionMintingService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

class PaymentWebhookServiceTest {
    @Test
    fun `a succeeded delivery for a refunded order is acknowledged and leaves the order refunded`() {
        val paidAt = OffsetDateTime.parse("2026-10-01T00:00:00Z")
        val order = Order(
            eventId = UUID.randomUUID(),
            recipientEmail = "buyer@test.local",
            paymentMethodWire = "gcash",
            status = OrderStatus.REFUNDED,
            totalPhp = BigDecimal("125.00"),
            paidAt = paidAt,
        )
        val orders = Mockito.mock(OrderRepository::class.java)
        val grants = Mockito.mock(DownloadGrantRepository::class.java)
        val ledger = Mockito.mock(TransactionMintingService::class.java)
        val removal = Mockito.mock(OrderPhotoRemovalService::class.java)
        Mockito.`when`(orders.findById(order.id)).thenReturn(Optional.of(order))
        val service = PaymentWebhookService(
            orders, Mockito.mock(OrderItemRepository::class.java), Mockito.mock(PaymentRepository::class.java),
            grants, ledger, removal,
        )

        val response = service.handle("stub", PaymentWebhookRequest(order.id, "synthetic_paid"))

        assertEquals("REFUNDED", response.orderStatus)
        assertEquals(0, response.grantsMinted)
        assertEquals(OrderStatus.REFUNDED, order.status)
        assertEquals(paidAt, order.paidAt)
        Mockito.verify(orders, Mockito.never()).save(order)
        Mockito.verify(grants, Mockito.never()).save(Mockito.any())
        Mockito.verifyNoInteractions(ledger, removal)
    }
}
