package com.quickpitik.service.orders

import com.quickpitik.dto.orders.PaymentWebhookRequest
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.repository.*
import com.quickpitik.service.earnings.TransactionMintingService
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import java.math.BigDecimal
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

class PaymentWebhookRemovalLockTest {
    private fun <T> arg(): T = any()
    @Test fun `stub settlement locks photos before its order row and ledger writes`() {
        val order = Order(eventId = UUID.randomUUID(), recipientEmail = "buyer@test.local", paymentMethodWire = "gcash",
            totalPhp = BigDecimal("125.00"))
        val orders = mock(OrderRepository::class.java)
        val payments = mock(PaymentRepository::class.java)
        val removal = mock(OrderPhotoRemovalService::class.java)
        val ledger = mock(TransactionMintingService::class.java)
        `when`(orders.findById(order.id)).thenReturn(Optional.of(order))
        val service = PaymentWebhookService(orders, mock(OrderItemRepository::class.java), payments,
            mock(DownloadGrantRepository::class.java), ledger, removal)
        service.handle("stub", PaymentWebhookRequest(order.id, "synthetic_paid"))
        val sequence = inOrder(removal, orders, ledger)
        sequence.verify(removal).lockSettlementPhotos(listOf(order.id))
        sequence.verify(orders).save(order)
        sequence.verify(removal).onOrderSettled(eq(order.id) ?: order.id, arg())
        sequence.verify(ledger).mintForPaidOrder(order.id)
        assertEquals(OrderStatus.PAID, order.status)
    }
}
