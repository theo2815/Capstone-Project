package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

class OrderReceiptRepositoryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var orders: OrderRepository

    @Autowired
    private lateinit var events: EventRepository

    @Autowired
    private lateinit var transactionTemplate: TransactionTemplate

    @Test
    fun `delivery stamping persists independently from the send claim`() {
        val order = order(emailSentAt = OLD_CLAIM)

        val updated = transactionTemplate.execute {
            orders.stampReceiptDelivered(order.id, DELIVERED_AT)
        }

        assertEquals(1, updated)
        assertEquals(DELIVERED_AT, orders.findById(order.id).orElseThrow().receiptDeliveredAt)
    }

    @Test
    fun `stale receipt lookup returns only the oldest fifty undelivered claims`() {
        val eligible = (0..50).map { index ->
            order(emailSentAt = OLD_CLAIM.plusSeconds(index.toLong()))
        }
        order(emailSentAt = CUTOFF.plusSeconds(1))
        order(emailSentAt = OLD_CLAIM, receiptDeliveredAt = DELIVERED_AT)
        order(emailSentAt = null)

        val found = orders.findStaleReceiptClaimIds(CUTOFF, PageRequest.of(0, 50))

        assertEquals(eligible.take(50).map { it.id }, found)
    }

    @Test
    fun `conditional stale release cannot clear a fresh reclaim or delivered claim`() {
        val order = order(emailSentAt = OLD_CLAIM)
        assertEquals(1, transactionTemplate.execute {
            orders.releaseStaleReceiptSend(order.id, CUTOFF)
        })
        assertEquals(1, transactionTemplate.execute {
            orders.claimReceiptSend(order.id, FRESH_CLAIM)
        })

        assertEquals(0, transactionTemplate.execute {
            orders.releaseStaleReceiptSend(order.id, CUTOFF)
        })
        assertEquals(FRESH_CLAIM, orders.findById(order.id).orElseThrow().emailSentAt)

        assertEquals(1, transactionTemplate.execute {
            orders.stampReceiptDelivered(order.id, DELIVERED_AT)
        })
        assertEquals(0, transactionTemplate.execute {
            orders.releaseStaleReceiptSend(order.id, FRESH_CLAIM.plusMinutes(1))
        })
        assertEquals(FRESH_CLAIM, orders.findById(order.id).orElseThrow().emailSentAt)
    }

    private fun order(
        emailSentAt: OffsetDateTime?,
        receiptDeliveredAt: OffsetDateTime? = null,
    ): Order = orders.saveAndFlush(
        Order(
            eventId = eventId,
            recipientEmail = "receipt-${UUID.randomUUID()}@test.local",
            paymentMethodWire = "gcash",
            status = OrderStatus.PAID,
            totalPhp = BigDecimal("125.00"),
            emailSentAt = emailSentAt,
            receiptDeliveredAt = receiptDeliveredAt,
        ),
    )

    private val eventId: UUID by lazy {
        events.saveAndFlush(
            Event(
                slug = "receipt-${UUID.randomUUID()}",
                name = "Receipt Test",
                date = LocalDate.of(2026, 9, 16),
                location = "Cebu City",
                status = EventStatus.COMPLETED,
            ),
        ).id
    }

    private companion object {
        val OLD_CLAIM: OffsetDateTime = OffsetDateTime.parse("2000-01-01T00:00:00Z")
        val CUTOFF: OffsetDateTime = OffsetDateTime.parse("2001-01-01T00:00:00Z")
        val FRESH_CLAIM: OffsetDateTime = OffsetDateTime.parse("2026-09-16T12:00:00Z")
        val DELIVERED_AT: OffsetDateTime = OffsetDateTime.parse("2026-09-16T12:01:00Z")
    }
}
