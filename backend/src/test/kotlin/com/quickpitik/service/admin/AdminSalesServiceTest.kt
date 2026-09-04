package com.quickpitik.service.admin

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PlatformProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
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

// The admin dashboard reconstructs gross from what photographers kept. Once a
// coupon exists, kept alone under-reports the platform fee — the discount has
// to be added back before dividing by the keep rate. Worked example: ₱150
// list, PHOTO20 → kept 90.00, discount 22.50, platform fee still 37.50.
class AdminSalesServiceTest {
    private lateinit var transactions: TransactionRepository
    private lateinit var eventPhotographers: EventPhotographerRepository
    private lateinit var events: EventRepository
    private lateinit var service: AdminSalesService

    @BeforeEach
    fun setUp() {
        transactions = Mockito.mock(TransactionRepository::class.java)
        eventPhotographers = Mockito.mock(EventPhotographerRepository::class.java)
        events = Mockito.mock(EventRepository::class.java)
        service = AdminSalesService(transactions, eventPhotographers, events, PlatformProperties())
    }

    @Test
    fun `platform revenue counts the full cut on a coupon sale`() {
        Mockito.`when`(transactions.sumPhotographerKeepInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("90.00"))
        Mockito.`when`(transactions.sumDiscountsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal("22.50"))
        Mockito.`when`(transactions.sumRefundsInWindow(anyArg(), anyArg())).thenReturn(BigDecimal.ZERO)
        Mockito.`when`(transactions.countSalesInWindow(anyArg(), anyArg())).thenReturn(1L)

        val dto = service.kpis(null)

        assertEquals(BigDecimal("150.00"), dto.gmv)
        assertEquals(BigDecimal("37.50"), dto.platformRevenue)
        assertEquals(BigDecimal("90.00"), dto.photographerKeep)
    }

    @Test
    fun `per-event implied gross adds the discount back before dividing by the keep rate`() {
        val event = Event(
            id = UUID.randomUUID(),
            slug = "cebu-marathon",
            name = "Cebu Marathon",
            date = LocalDate.of(2026, 8, 29),
            location = "Cebu City",
            status = EventStatus.ACTIVE,
        )
        Mockito.`when`(eventPhotographers.salesAggregatesByEvent()).thenReturn(
            listOf(arrayOf<Any>(event.id, BigDecimal("90.00"), BigDecimal.ZERO, 1L, BigDecimal("22.50"))),
        )
        Mockito.`when`(events.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event))

        val row = service.byEvent(null, PaginationParams.of(0, 20)).items.single()

        assertEquals(BigDecimal("150.00"), row.impliedGmv)
        assertEquals(BigDecimal("90.00"), row.impliedCut)
    }

    private fun <T> anyArg(): T = Mockito.any()
}
