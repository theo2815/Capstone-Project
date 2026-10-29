package com.quickpitik.service.orders

import com.quickpitik.entity.Order
import com.quickpitik.repository.OrderRepository
import com.quickpitik.service.email.EmailService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.util.Optional
import java.util.UUID

class RefundLinkSentEmailListenerTest {
    private val orders = Mockito.mock(OrderRepository::class.java)
    private val emailService = Mockito.mock(EmailService::class.java)
    private val listener = RefundLinkSentEmailListener(orders, emailService)

    @Test
    fun `mails the link to the order's recipient email`() {
        val order = Order(eventId = UUID.randomUUID(), recipientEmail = "runner@example.com", paymentMethodWire = "qrph", totalPhp = BigDecimal("127.50"))
        Mockito.`when`(orders.findById(order.id)).thenReturn(Optional.of(order))

        listener.onRefundLinkSent(RefundLinkSentEvent(order.id, "https://transfer.test/abc", BigDecimal("127.50")))

        Mockito.verify(emailService).sendRefundLinkEmail("runner@example.com", "https://transfer.test/abc", BigDecimal("127.50"))
    }

    @Test
    fun `sends nothing when the order no longer exists`() {
        val orderId = UUID.randomUUID()
        Mockito.`when`(orders.findById(orderId)).thenReturn(Optional.empty())

        listener.onRefundLinkSent(RefundLinkSentEvent(orderId, "https://transfer.test/abc", BigDecimal("127.50")))

        Mockito.verifyNoInteractions(emailService)
    }
}
