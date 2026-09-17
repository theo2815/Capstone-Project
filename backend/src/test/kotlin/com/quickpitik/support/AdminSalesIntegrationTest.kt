package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeReason
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.Transaction
import com.quickpitik.entity.User
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminSalesService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals

class AdminSalesIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var service: AdminSalesService
    @Autowired private lateinit var disputes: DisputeRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var orderItems: OrderItemRepository
    @Autowired private lateinit var orders: OrderRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var transactions: TransactionRepository
    @Autowired private lateinit var users: UserRepository

    @Test
    fun `admin sales uses succeeded dispute amounts for refunds`() {
        val before = service.kpis(null).refundsIssued
        val now = OffsetDateTime.now(ZoneOffset.ofHours(8))
        val photographer = users.saveAndFlush(
            User(
                email = "admin-sales-${UUID.randomUUID()}@test.local",
                passwordHash = "unused",
                name = "Admin Sales",
                role = Role.PHOTOGRAPHER,
            ),
        )
        val event = event()
        val photo = photo(event.id, photographer.id)
        val order = order(event.id)
        val sale = transactions.saveAndFlush(
            Transaction(
                paidAt = now,
                photographerId = photographer.id,
                eventId = event.id,
                photoId = photo.id,
                orderId = order.id,
                amountKeptPhp = BigDecimal("62.81"),
            ),
        )
        transactions.saveAndFlush(
            Transaction(
                paidAt = now,
                photographerId = photographer.id,
                eventId = event.id,
                photoId = photo.id,
                orderId = order.id,
                amountKeptPhp = BigDecimal("-62.81"),
                isRefund = true,
                refundOf = sale.id,
            ),
        )
        dispute(order.id, photo.id, "94.06", "succeeded", now)
        dispute(order.id, photo.id, "50.00", "failed", now)

        val oldEvent = event()
        val oldPhoto = photo(oldEvent.id, photographer.id)
        val oldOrder = order(oldEvent.id)
        dispute(oldOrder.id, oldPhoto.id, "60.00", "succeeded", now.minusYears(2))

        val after = service.kpis(null).refundsIssued
        val eventRow = service.byEvent(null, PaginationParams.of(0, 100)).items
            .single { it.id == event.id.toString() }

        assertEquals(BigDecimal("94.06"), after.subtract(before))
        assertEquals(BigDecimal("94.06"), eventRow.refundsIssued)
    }

    // M-16: a ledger minted at an older 80 % keep. Dividing by today's 75 %
    // would report 226.67; GMV is the list price the items sold at, less the
    // listed share a refund reversed (half of the ₱125 item, all of the ₱100).
    // Per-event GMV counts sales only; refunds show in their own column.
    @Test
    fun `admin gmv is the order items' list price, net of refunded shares`() {
        val before = service.kpis(null).gmv
        val now = OffsetDateTime.now(ZoneOffset.ofHours(8))
        val photographer = users.saveAndFlush(
            User(
                email = "admin-sales-${UUID.randomUUID()}@test.local",
                passwordHash = "unused",
                name = "Admin Sales",
                role = Role.PHOTOGRAPHER,
            ),
        )
        val event = event()
        val order = order(event.id)
        val plain = photo(event.id, photographer.id)
        val couponed = photo(event.id, photographer.id)
        val returned = photo(event.id, photographer.id)
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, plain.id), BigDecimal("125.00")))
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, returned.id), BigDecimal("100.00")))
        orderItems.saveAndFlush(
            OrderItem(OrderItemId(order.id, couponed.id), BigDecimal("150.00"), discountPhp = BigDecimal("22.50")),
        )
        val sale = transactions.saveAndFlush(sale(now, photographer.id, event.id, plain.id, order.id, "100.00", "0"))
        transactions.saveAndFlush(sale(now, photographer.id, event.id, couponed.id, order.id, "97.50", "22.50"))
        val returnedSale = transactions.saveAndFlush(
            sale(now, photographer.id, event.id, returned.id, order.id, "80.00", "0"),
        )
        transactions.saveAndFlush(refund(sale, "-50.00"))
        transactions.saveAndFlush(refund(returnedSale, "-80.00"))

        val after = service.kpis(null).gmv
        val eventRow = service.byEvent(null, PaginationParams.of(0, 100)).items
            .single { it.id == event.id.toString() }

        assertEquals(BigDecimal("212.50"), after.subtract(before))
        assertEquals(BigDecimal("375.00"), eventRow.impliedGmv)
        assertEquals(BigDecimal("75.00"), eventRow.impliedCut)
    }

    private fun refund(sale: Transaction, kept: String) = Transaction(
        paidAt = sale.paidAt,
        photographerId = sale.photographerId,
        eventId = sale.eventId,
        photoId = sale.photoId,
        orderId = sale.orderId,
        amountKeptPhp = BigDecimal(kept),
        isRefund = true,
        refundOf = sale.id,
    )

    private fun sale(
        paidAt: OffsetDateTime,
        photographerId: UUID,
        eventId: UUID,
        photoId: UUID,
        orderId: UUID,
        kept: String,
        discount: String,
    ) = Transaction(
        paidAt = paidAt,
        photographerId = photographerId,
        eventId = eventId,
        photoId = photoId,
        orderId = orderId,
        amountKeptPhp = BigDecimal(kept),
        discountPhp = BigDecimal(discount),
    )

    private fun event(): Event = events.saveAndFlush(
        Event(
            slug = "admin-sales-${UUID.randomUUID()}",
            name = "Admin Sales",
            date = LocalDate.now(),
            location = "Cebu City",
            status = EventStatus.ACTIVE,
        ),
    )

    private fun photo(eventId: UUID, photographerId: UUID): Photo = photos.saveAndFlush(
        Photo(
            eventId = eventId,
            photographerId = photographerId,
            s3Key = "admin-sales/${UUID.randomUUID()}.jpg",
            pricePhp = BigDecimal("125.00"),
        ),
    )

    private fun order(eventId: UUID): Order = orders.saveAndFlush(
        Order(
            eventId = eventId,
            recipientEmail = "buyer@test.local",
            paymentMethodWire = "gcash",
            status = OrderStatus.REFUNDED,
            totalPhp = BigDecimal("94.06"),
        ),
    )

    private fun dispute(
        orderId: UUID,
        photoId: UUID,
        amount: String,
        refundStatus: String,
        refundedAt: OffsetDateTime,
    ) {
        disputes.saveAndFlush(
            Dispute(
                orderId = orderId,
                photoId = photoId,
                reasonWire = DisputeReason.OTHER.wire,
                statusWire = DisputeStatus.RESOLVED.wire,
                resolutionWire = DisputeResolution.REFUND_FULL.wire,
                refundAmountPhp = BigDecimal(amount),
                refundStatus = refundStatus,
                refundedAt = refundedAt,
            ),
        )
    }
}
