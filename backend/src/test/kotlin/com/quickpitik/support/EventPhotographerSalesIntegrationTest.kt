package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.earnings.EarningsService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

class EventPhotographerSalesIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var eventPhotographers: EventPhotographerRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var orders: OrderRepository
    @Autowired private lateinit var orderItems: OrderItemRepository
    @Autowired private lateinit var earnings: EarningsService
    @Autowired private lateinit var transactions: TransactionTemplate

    private lateinit var key: EventPhotographerId

    @BeforeEach
    fun seed() {
        val photographer = users.save(
            User(
                email = "sales-${UUID.randomUUID()}@test.local",
                passwordHash = "unused",
                name = "Sales",
                role = Role.PHOTOGRAPHER,
            ),
        )
        val event = events.save(
            Event(
                slug = "sales-${UUID.randomUUID()}",
                name = "Sales",
                date = LocalDate.now(),
                location = "Cebu",
                status = EventStatus.ACTIVE,
            ),
        )
        key = EventPhotographerId(event.id, photographer.id)
    }

    @Test
    fun `first sale inserts the event photographer counters`() {
        adjust(1, BigDecimal("70.54"))

        val row = eventPhotographers.findById(key).orElseThrow()
        assertEquals(1, row.salesCount)
        assertEquals(BigDecimal("70.54"), row.revenueKeptPhp)
    }

    @Test
    fun `two sales atomically add both counter deltas`() {
        adjust(1, BigDecimal("70.54"))
        adjust(1, BigDecimal("23.52"))

        val row = eventPhotographers.findById(key).orElseThrow()
        assertEquals(2, row.salesCount)
        assertEquals(BigDecimal("94.06"), row.revenueKeptPhp)
    }

    @Test
    fun `a full refund returns sale count and revenue to zero`() {
        adjust(1, BigDecimal("70.54"))
        adjust(-1, BigDecimal("-70.54"))

        val row = eventPhotographers.findById(key).orElseThrow()
        assertEquals(0, row.salesCount)
        assertEquals(BigDecimal("0.00"), row.revenueKeptPhp)
    }

    @Test
    fun `subtracting more sales than recorded floors the count at zero`() {
        adjust(1, BigDecimal("70.54"))
        adjust(-2, BigDecimal("-70.54"))

        val row = eventPhotographers.findById(key).orElseThrow()
        assertEquals(0, row.salesCount)
        assertEquals(BigDecimal("0.00"), row.revenueKeptPhp)
    }

    @Test
    fun `purchase membership and counts exclude other sellers`() {
        val seller = users.save(User(email = "own-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Own", role = Role.PHOTOGRAPHER))
        val other = users.save(User(email = "other-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Other", role = Role.PHOTOGRAPHER))
        val paid = events.save(event("Paid"))
        val free = events.save(event("Free"))
        val refunded = events.save(event("Refunded"))
        val empty = events.save(event("Empty"))
        val foreignOnly = events.save(event("Foreign only"))
        eventPhotographers.save(EventPhotographer(EventPhotographerId(paid.id, seller.id), revenueKeptPhp = BigDecimal("75.00")))
        listOf(free, refunded, empty, foreignOnly).forEach {
            eventPhotographers.save(EventPhotographer(EventPhotographerId(it.id, seller.id)))
        }
        val ownPhotos = listOf(paid, free, refunded).associateWith {
            photos.save(Photo(eventId = it.id, photographerId = seller.id, s3Key = "own/${it.id}.jpg", pricePhp = BigDecimal("100.00")))
        }
        val otherPhoto = photos.save(Photo(eventId = paid.id, photographerId = other.id, s3Key = "other.jpg", pricePhp = BigDecimal("100.00")))
        val foreignOnlyPhoto = photos.save(Photo(eventId = foreignOnly.id, photographerId = other.id, s3Key = "foreign-only.jpg", pricePhp = BigDecimal("100.00")))
        fun purchase(event: Event, status: OrderStatus, discount: BigDecimal = BigDecimal.ZERO): Order {
            val price = BigDecimal("100.00")
            val order = orders.save(Order(eventId = event.id, recipientEmail = "buyer@test.local", paymentMethodWire = "gcash", status = status, totalPhp = price.subtract(discount), paidAt = OffsetDateTime.parse("2026-09-01T10:00:00Z")))
            orderItems.save(OrderItem(OrderItemId(order.id, ownPhotos.getValue(event).id), price, discount))
            return order
        }
        val paidOrder = purchase(paid, OrderStatus.PAID)
        orderItems.save(OrderItem(OrderItemId(paidOrder.id, otherPhoto.id), BigDecimal("100.00")))
        purchase(free, OrderStatus.FULFILLED, BigDecimal("100.00"))
        purchase(refunded, OrderStatus.REFUNDED)
        val foreignOrder = orders.save(Order(eventId = foreignOnly.id, recipientEmail = "buyer@test.local", paymentMethodWire = "gcash", status = OrderStatus.FULFILLED, totalPhp = BigDecimal("100.00"), paidAt = OffsetDateTime.parse("2026-09-01T10:00:00Z")))
        orderItems.save(OrderItem(OrderItemId(foreignOrder.id, foreignOnlyPhoto.id), BigDecimal("100.00")))

        val page = eventPhotographers.pageEarningsForPhotographer(seller.id, org.springframework.data.domain.PageRequest.of(0, 200))
        val counts = orderItems.countPurchasesByEventIds(seller.id, listOf(paid.id, free.id, refunded.id, foreignOnly.id))
        val apiRows = earnings.listPerEvent(seller.id, PaginationParams(0, 200)).items.associateBy { UUID.fromString(it.eventId) }

        assertEquals(1L, apiRows[free.id]?.purchaseCount, "giveaway-only event must remain visible")
        assertEquals(0, apiRows.getValue(free.id).revenueKept.compareTo(BigDecimal.ZERO))
        assertEquals(
            emptySet(),
            buildSet {
                if (page.content.any { it.id.eventId == foreignOnly.id }) add("membership")
                if (counts.any { it.eventId == foreignOnly.id }) add("purchase count")
            },
            "foreign-only event must not appear in seller earnings",
        )
        assertEquals(setOf(paid.id, free.id, refunded.id), page.content.map { it.id.eventId }.toSet())
        assertEquals(mapOf(paid.id to 1L, free.id to 1L, refunded.id to 1L), counts.associate { it.eventId to it.purchaseCount })
    }

    @Test
    fun `earnings pages beyond two hundred eligible events`() {
        val seller = users.save(User(email = "paged-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Paged", role = Role.PHOTOGRAPHER))
        val eligible = events.saveAll((0..200).map { event("Paged $it") })
        eventPhotographers.saveAll(eligible.map {
            EventPhotographer(EventPhotographerId(it.id, seller.id), revenueKeptPhp = BigDecimal.ONE)
        })

        val first = eventPhotographers.pageEarningsForPhotographer(seller.id, org.springframework.data.domain.PageRequest.of(0, 200))
        val second = eventPhotographers.pageEarningsForPhotographer(seller.id, org.springframework.data.domain.PageRequest.of(1, 200))

        assertEquals(201, first.totalElements)
        assertEquals(200, first.content.size)
        assertEquals(1, second.content.size)
        assertEquals(201, (first.content + second.content).map { it.id.eventId }.toSet().size)
    }

    private fun adjust(saleDelta: Int, keptDelta: BigDecimal) {
        transactions.executeWithoutResult {
            eventPhotographers.adjustSalesCounters(
                key.eventId,
                key.photographerId,
                saleDelta,
                keptDelta,
            )
        }
    }

    private fun event(name: String) = Event(
        slug = "${name.lowercase().replace(' ', '-')}-${UUID.randomUUID()}",
        name = name,
        date = LocalDate.of(2026, 9, 1),
        location = "Cebu",
        status = EventStatus.ACTIVE,
    )
}
