package com.quickpitik.support

import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Pins DisputeRepository.findDueForAutoCancel: which parked QR Ph refunds the hourly auto-cancel job is shown.
// The 30-day and 3-day cut-offs are computed by the service, so here they are plain timestamps.
class DisputeAutoCancelQueryIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var disputes: DisputeRepository

    private val now = OffsetDateTime.parse("2026-11-30T00:00:00Z")
    private val windowCutoff = now.minusDays(30) // Oct 31
    private val linkCutoff = now.minusDays(3) // Nov 27
    private val longAgo = OffsetDateTime.parse("2026-10-01T00:00:00Z")

    private fun dueIds(): Set<UUID> =
        disputes.findDueForAutoCancel(windowCutoff, linkCutoff, PageRequest.of(0, 500)).map { it.id }.toSet()

    @Test
    fun `a parked QR Ph refund past the window with no link is due`() {
        val d = parked(paidAt = longAgo)

        assertTrue(d.id in dueIds())
    }

    @Test
    fun `a refund whose window has not passed is not due`() {
        val d = parked(paidAt = now.minusDays(5))

        assertFalse(d.id in dueIds())
    }

    @Test
    fun `a link sent less than three days ago keeps the refund out and an older link does not`() {
        val fresh = parked(paidAt = longAgo, linkSentAt = now.minusDays(1))
        val stale = parked(paidAt = longAgo, linkSentAt = now.minusDays(4))

        val due = dueIds()

        assertFalse(fresh.id in due)
        assertTrue(stale.id in due)
    }

    @Test
    fun `only QR Ph payments are due`() {
        val gcash = parked(paidAt = longAgo, method = "gcash")

        assertFalse(gcash.id in dueIds())
    }

    @Test
    fun `a refund that is not waiting for a manual refund is not due`() {
        val ids = listOf("succeeded", "cancelled", "requesting").map { parked(paidAt = longAgo, refundStatus = it).id }

        assertTrue(dueIds().none { it in ids })
    }

    @Test
    fun `a refund with no approving admin is not due`() {
        val d = parked(paidAt = longAgo, approvedBy = false)

        assertFalse(d.id in dueIds())
    }

    @Test
    fun `an order with no paid time counts from when it was created`() {
        val old = parked(paidAt = null, createdAt = longAgo)
        val recent = parked(paidAt = null, createdAt = now.minusDays(5))
        val paidRecentlyButCreatedLongAgo = parked(paidAt = now.minusDays(5), createdAt = longAgo)

        val due = dueIds()

        assertTrue(old.id in due)
        assertFalse(recent.id in due)
        assertFalse(paidRecentlyButCreatedLongAgo.id in due)
    }

    @Test
    fun `the oldest approved refund comes first`() {
        val newer = parked(paidAt = longAgo, requestedAt = longAgo.plusDays(2))
        val older = parked(paidAt = longAgo, requestedAt = longAgo.plusDays(1))

        val ids = disputes.findDueForAutoCancel(windowCutoff, linkCutoff, PageRequest.of(0, 500)).map { it.id }

        assertTrue(ids.indexOf(older.id) in 0 until ids.indexOf(newer.id))
    }

    private fun parked(
        paidAt: OffsetDateTime?,
        createdAt: OffsetDateTime = longAgo,
        linkSentAt: OffsetDateTime? = null,
        method: String = "qrph",
        refundStatus: String = "manual_review",
        approvedBy: Boolean = true,
        requestedAt: OffsetDateTime = longAgo.plusDays(1),
    ): Dispute {
        val runner = user(Role.RUNNER)
        val admin = user(Role.ADMIN)
        val photographer = user(Role.PHOTOGRAPHER)
        val event = events.saveAndFlush(
            Event(
                slug = "auto-cancel-${UUID.randomUUID()}",
                name = "Auto-cancel fixture",
                date = LocalDate.of(2026, 9, 15),
                location = "Cebu",
                status = EventStatus.ACTIVE,
            ),
        )
        val photo = photos.saveAndFlush(
            Photo(
                eventId = event.id,
                photographerId = photographer.id,
                s3Key = "auto-cancel/${UUID.randomUUID()}.jpg",
                pricePhp = BigDecimal("100.00"),
            ),
        )
        val order = orders.saveAndFlush(
            Order(
                userId = runner.id,
                eventId = event.id,
                recipientEmail = runner.email,
                paymentMethodWire = method,
                status = OrderStatus.FULFILLED,
                totalPhp = BigDecimal("100.00"),
                paidAt = paidAt,
                createdAt = createdAt,
            ),
        )
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("100.00")))
        return disputes.saveAndFlush(
            Dispute(
                orderId = order.id,
                photoId = photo.id,
                runnerId = runner.id,
                reasonWire = "other",
                statusWire = DisputeStatus.ESCALATED.wire,
            ).apply {
                this.refundStatus = refundStatus
                refundAmountPhp = BigDecimal("100.00")
                refundRequestedAt = requestedAt
                refundRequestedBy = if (approvedBy) admin.id else null
                refundLinkSentAt = linkSentAt
            },
        )
    }

    private fun user(role: Role) = users.saveAndFlush(
        User(
            email = "auto-cancel-${UUID.randomUUID()}@test.local",
            passwordHash = "test-only-unused",
            name = "Auto-cancel fixture",
            role = role,
        ),
    )
}
