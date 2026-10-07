package com.quickpitik.service.earnings

import com.quickpitik.common.PaginationParams
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventPurchaseCountProjection
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.TransactionRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

class EarningsServiceTest {
    private val transactions = Mockito.mock(TransactionRepository::class.java)
    private val payouts = Mockito.mock(PayoutCycleRepository::class.java)
    private val events = Mockito.mock(EventRepository::class.java)
    private val coverage = Mockito.mock(EventPhotographerRepository::class.java)
    private val purchases = Mockito.mock(OrderItemRepository::class.java)
    private val service = EarningsService(transactions, payouts, events, coverage, purchases)
    private val seller = UUID.randomUUID()

    @Test
    fun `empty out-of-range page retains the repository total and requested window`() {
        val params = PaginationParams(250, 50)
        Mockito.`when`(coverage.pageEarningsForPhotographer(anyArg(), anyArg()))
            .thenReturn(PageImpl(emptyList(), OffsetLimitPageable(params), 205))

        val result = service.listPerEvent(seller, params)

        assertEquals(emptyList(), result.items)
        assertEquals(205L, result.total)
        assertEquals(250, result.offset)
        assertEquals(50, result.limit)
        Mockito.verifyNoInteractions(events, purchases, transactions, payouts)
    }

    @Test
    fun `zero history has zero total and echoes the requested window`() {
        val params = PaginationParams(250, 50)
        Mockito.`when`(coverage.pageEarningsForPhotographer(anyArg(), anyArg()))
            .thenReturn(PageImpl(emptyList(), OffsetLimitPageable(params), 0))

        val result = service.listPerEvent(seller, params)

        assertEquals(emptyList(), result.items)
        assertEquals(0L, result.total)
        assertEquals(250, result.offset)
        assertEquals(50, result.limit)
    }

    @Test
    fun `includes giveaway-only and fully refunded events`() {
        val paid = event("Paid")
        val free = event("Giveaway")
        val refunded = event("Refunded", OffsetDateTime.parse("2026-09-04T21:36:27Z"))
        val rows = listOf(
            coverage(paid.id, "75.00"),
            coverage(free.id, "0.00"),
            coverage(refunded.id, "0.00"),
        )
        val counts = listOf(count(paid.id, 1), count(free.id, 2), count(refunded.id, 1))
        Mockito.`when`(coverage.pageEarningsForPhotographer(anyArg(), anyArg()))
            .thenReturn(PageImpl(rows))
        Mockito.`when`(events.findAllById(rows.map { it.id.eventId }))
            .thenReturn(listOf(paid, free, refunded))
        Mockito.`when`(purchases.countPurchasesByEventIds(seller, rows.map { it.id.eventId }))
            .thenReturn(counts)

        val result = service.listPerEvent(seller, PaginationParams(0, 200))

        assertEquals(setOf(paid.id, free.id, refunded.id), result.items.map { UUID.fromString(it.eventId) }.toSet())
        assertEquals(mapOf(paid.id to 1L, free.id to 2L, refunded.id to 1L), result.items.associate { UUID.fromString(it.eventId) to it.purchaseCount })
        assertEquals(refunded.deletedAt, result.items.single { it.eventId == refunded.id.toString() }.deletedAt)
        Mockito.verify(purchases).countPurchasesByEventIds(seller, rows.map { it.id.eventId })
    }

    @Test
    fun `keeps positive revenue legacy rows without receipts`() {
        val legacy = event("Legacy")
        val row = coverage(legacy.id, "40.00")
        Mockito.`when`(coverage.pageEarningsForPhotographer(anyArg(), anyArg()))
            .thenReturn(PageImpl(listOf(row)))
        Mockito.`when`(events.findAllById(listOf(legacy.id))).thenReturn(listOf(legacy))
        Mockito.`when`(purchases.countPurchasesByEventIds(seller, listOf(legacy.id))).thenReturn(emptyList())

        val result = service.listPerEvent(seller, PaginationParams(0, 20))

        assertEquals(legacy.id.toString(), result.items.single().eventId)
        assertEquals(0L, result.items.single().purchaseCount)
        assertEquals(BigDecimal("40.00"), result.items.single().revenueKept)
    }

    private fun event(name: String, deletedAt: OffsetDateTime? = null) = Event(
        slug = "${name.lowercase()}-${UUID.randomUUID()}",
        name = name,
        date = LocalDate.of(2026, 9, 1),
        location = "Cebu",
        status = EventStatus.ACTIVE,
        deletedAt = deletedAt,
    )

    private fun coverage(eventId: UUID, revenue: String) = EventPhotographer(
        EventPhotographerId(eventId, seller),
        revenueKeptPhp = BigDecimal(revenue),
    )

    private fun count(eventId: UUID, value: Long): EventPurchaseCountProjection =
        Mockito.mock(EventPurchaseCountProjection::class.java).also {
            Mockito.`when`(it.eventId).thenReturn(eventId)
            Mockito.`when`(it.purchaseCount).thenReturn(value)
        }

    private fun <T> anyArg(): T = Mockito.any()
}
