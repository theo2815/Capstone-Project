package com.quickpitik.service.admin

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.TransactionRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals

// GMV is the list price of what sold (order_items); the platform fee is
// GMV − (kept + coupon discount the photographer absorbed). Worked
// example: ₱150 list, PHOTO20 → kept 90.00, discount 22.50, platform fee still 37.50.
class AdminSalesServiceTest {
    private lateinit var transactions: TransactionRepository
    private lateinit var disputes: DisputeRepository
    private lateinit var eventPhotographers: EventPhotographerRepository
    private lateinit var events: EventRepository
    private lateinit var service: AdminSalesService

    @BeforeEach
    fun setUp() {
        transactions = Mockito.mock(TransactionRepository::class.java)
        disputes = Mockito.mock(DisputeRepository::class.java)
        eventPhotographers = Mockito.mock(EventPhotographerRepository::class.java)
        events = Mockito.mock(EventRepository::class.java)
        service = AdminSalesService(transactions, disputes, eventPhotographers, events)
    }

    @Test
    fun `platform revenue counts the full cut on a coupon sale`() {
        Mockito.`when`(transactions.sumPhotographerKeepInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("90.00"))
        Mockito.`when`(transactions.sumDiscountsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("22.50"))
        Mockito.`when`(transactions.sumListGrossInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("150.00"))
        Mockito.`when`(disputes.sumSucceededRefundsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal.ZERO)
        Mockito.`when`(transactions.countSalesInWindow(anyArg(), anyArg())).thenReturn(1L)

        val dto = service.kpis(null)

        assertEquals(BigDecimal("150.00"), dto.gmv)
        assertEquals(BigDecimal("37.50"), dto.platformRevenue)
        assertEquals(BigDecimal("90.00"), dto.photographerKeep)
    }

    @Test
    fun `net platform revenue equals the already-net platform revenue`() {
        Mockito.`when`(transactions.sumPhotographerKeepInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("75.00"))
        Mockito.`when`(transactions.sumDiscountsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal.ZERO)
        Mockito.`when`(transactions.sumListGrossInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("100.00"))
        Mockito.`when`(disputes.sumSucceededRefundsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("100.00"))
        Mockito.`when`(transactions.countSalesInWindow(anyArg(), anyArg())).thenReturn(1L)

        val dto = service.kpis(null)

        assertEquals(BigDecimal("25.00"), dto.platformRevenue)
        assertEquals(dto.platformRevenue, dto.netPlatformRevenue)
    }

    @Test
    fun `refunds issued is the peso amount credited to the buyer`() {
        Mockito.`when`(transactions.sumPhotographerKeepInWindow(anyArg(), anyArg())).thenReturn(BigDecimal.ZERO)
        Mockito.`when`(transactions.sumDiscountsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal.ZERO)
        Mockito.`when`(transactions.sumListGrossInWindow(anyArg(), anyArg())).thenReturn(BigDecimal.ZERO)
        Mockito.`when`(disputes.sumSucceededRefundsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("94.06"))
        Mockito.`when`(transactions.countSalesInWindow(anyArg(), anyArg())).thenReturn(1L)

        val dto = service.kpis(null)

        assertEquals(BigDecimal("94.06"), dto.refundsIssued)
    }

    @Test
    fun `per-event gmv is the list gross of its sales`() {
        val event = event()
        Mockito.`when`(eventPhotographers.salesAggregatesByEvent()).thenReturn(
            listOf(arrayOf<Any>(event.id, BigDecimal("90.00"), 1L, BigDecimal("22.50"), BigDecimal("150.00"))),
        )
        Mockito.`when`(disputes.sumSucceededRefundsByEvent()).thenReturn(emptyList())
        Mockito.`when`(events.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event))

        val row = service.byEvent(null, PaginationParams.of(0, 20)).items.single()

        assertEquals(BigDecimal("150.00"), row.impliedGmv)
        assertEquals(BigDecimal("37.50"), row.impliedCut)
    }

    @Test
    fun `per-event cut is gross minus photographer keep and discounts`() {
        val event = event()
        Mockito.`when`(eventPhotographers.salesAggregatesByEvent()).thenReturn(
            listOf(arrayOf<Any>(event.id, BigDecimal("7500.00"), 1L, BigDecimal.ZERO, BigDecimal("10000.00"))),
        )
        Mockito.`when`(disputes.sumSucceededRefundsByEvent()).thenReturn(emptyList())
        Mockito.`when`(events.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event))

        val row = service.byEvent(null, PaginationParams.of(0, 20)).items.single()

        assertEquals(BigDecimal("10000.00"), row.impliedGmv)
        assertEquals(BigDecimal("2500.00"), row.impliedCut)
    }

    @Test
    fun `per-event refunds use the peso amount credited to the buyer`() {
        val event = event()
        Mockito.`when`(eventPhotographers.salesAggregatesByEvent()).thenReturn(
            listOf(arrayOf<Any>(event.id, BigDecimal("7500.00"), 1L, BigDecimal.ZERO, BigDecimal("10000.00"))),
        )
        Mockito.`when`(disputes.sumSucceededRefundsByEvent()).thenReturn(
            listOf(arrayOf<Any>(event.id, BigDecimal("94.06"))),
        )
        Mockito.`when`(events.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event))

        val row = service.byEvent(null, PaginationParams.of(0, 20)).items.single()

        assertEquals(BigDecimal("94.06"), row.refundsIssued)
    }

    private fun event(): Event = Event(
        id = UUID.randomUUID(),
        slug = "cebu-marathon",
        name = "Cebu Marathon",
        date = LocalDate.of(2026, 8, 29),
        location = "Cebu City",
        status = EventStatus.ACTIVE,
    )

    private fun <T> anyArg(): T = Mockito.any()
}
