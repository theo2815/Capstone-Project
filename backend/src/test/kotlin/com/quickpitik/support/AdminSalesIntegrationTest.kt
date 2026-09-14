package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeReason
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.Transaction
import com.quickpitik.entity.User
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
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
