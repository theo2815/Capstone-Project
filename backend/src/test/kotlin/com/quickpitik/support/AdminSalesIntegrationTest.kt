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
import kotlin.test.assertTrue

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

    // AX-12: the newest trend week is the same Manila week as kpis(range=week),
    // read from the same rows, so the two must agree to the centavo.
    @Test
    fun `the trend's current week equals kpis for this week`() {
        val now = OffsetDateTime.now(ZoneOffset.ofHours(8))
        val photographer = photographer()
        val event = event()
        val order = order(event.id)
        val photo = photo(event.id, photographer.id)
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("150.00"), discountPhp = BigDecimal("22.50")))
        transactions.saveAndFlush(sale(now, photographer.id, event.id, photo.id, order.id, "90.00", "22.50"))
        dispute(order.id, photo.id, "12.34", "succeeded", now)

        val week = service.kpis("week")
        val point = service.trend(12).weeks.last()

        assertEquals(12, service.trend(12).weeks.size)
        assertEquals(week.gmv, point.gmv)
        assertEquals(week.platformRevenue, point.platformRevenue)
        assertEquals(week.refundsIssued, point.refundsIssued)
        assertEquals(week.totalSalesCount, point.salesCount)
        assertTrue(point.salesCount >= 1)
    }

    // List gross per photographer, net of the share a refund reversed; the
    // amounts are large so these two outrank whatever other tests left behind.
    @Test
    fun `top photographers rank by list gross net of refunds`() {
        val now = OffsetDateTime.now(ZoneOffset.ofHours(8))
        val event = event()
        val order = order(event.id)
        val refunded = photographer()
        val steady = photographer()
        val big = photo(event.id, refunded.id)
        val small = photo(event.id, steady.id)
        val small2 = photo(event.id, steady.id)
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, big.id), BigDecimal("900000.00")))
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, small.id), BigDecimal("300000.00")))
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, small2.id), BigDecimal("300000.00")))
        val bigSale = transactions.saveAndFlush(sale(now, refunded.id, event.id, big.id, order.id, "675000.00", "0"))
        transactions.saveAndFlush(refund(bigSale, "-450000.00"))
        transactions.saveAndFlush(sale(now, steady.id, event.id, small.id, order.id, "225000.00", "0"))
        transactions.saveAndFlush(sale(now, steady.id, event.id, small2.id, order.id, "225000.00", "0"))

        val items = service.topPhotographers("ytd", 20).items
            .filter { it.userId == refunded.id.toString() || it.userId == steady.id.toString() }

        assertEquals(listOf(steady.id.toString(), refunded.id.toString()), items.map { it.userId })
        assertEquals(BigDecimal("600000.00"), items[0].gmv)
        assertEquals(2L, items[0].salesCount)
        assertEquals(BigDecimal("300000.00"), items[1].gmv)
        assertEquals(1L, items[1].salesCount)
        assertEquals("Admin Sales", items[1].displayName)
    }

    private fun photographer(): User = users.saveAndFlush(
        User(
            email = "admin-sales-${UUID.randomUUID()}@test.local",
            passwordHash = "unused",
            name = "Admin Sales",
            role = Role.PHOTOGRAPHER,
        ),
    )

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
