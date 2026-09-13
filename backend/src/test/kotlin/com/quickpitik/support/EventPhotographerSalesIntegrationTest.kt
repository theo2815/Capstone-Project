package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals

class EventPhotographerSalesIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var eventPhotographers: EventPhotographerRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var users: UserRepository
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
}
