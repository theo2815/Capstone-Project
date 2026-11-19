package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.repository.EventRepository
import com.quickpitik.service.events.EventDtoMapper
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

// AX-9: the admin events list filters search, dates and state in SQL, so `total` counts the
// filtered set, and countForAdmin agrees. The state predicates must match deriveAdminEventState.
class AdminEventListQueryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var eventRepository: EventRepository

    private val today = LocalDate.of(2026, 6, 15)
    private val from = LocalDate.of(1900, 1, 1)
    private val to = LocalDate.of(9999, 12, 31)

    @Test
    fun `state filter returns only that state and total counts it`() {
        val marker = "ax9-${UUID.randomUUID()}"
        val expected = mapOf(
            "upcoming" to setOf(save(marker, EventStatus.ACTIVE, today.plusDays(3)), save(marker, EventStatus.DRAFT, today.minusDays(9))),
            "live" to setOf(save(marker, EventStatus.ACTIVE, today)),
            "open" to setOf(save(marker, EventStatus.ACTIVE, today.minusDays(1)), save(marker, EventStatus.COMPLETED, today.plusDays(5))),
            "past" to setOf(save(marker, EventStatus.ARCHIVED, today.plusDays(1))),
        )
        save(marker, EventStatus.ACTIVE, today, deletedAt = OffsetDateTime.now())

        for ((state, ids) in expected) {
            val page = eventRepository.pageForAdmin(marker, from, to, state, today, PageRequest.of(0, 50))
            assertEquals(ids, page.content.map { it.id }.toSet(), "rows for $state")
            assertEquals(ids.size.toLong(), page.totalElements, "total for $state")
            assertEquals(ids.size.toLong(), eventRepository.countForAdmin(marker, from, to, state, today), "count for $state")
            page.content.forEach { assertEquals(state, EventDtoMapper.deriveAdminEventState(it, today), "mapper agrees for ${it.id}") }
        }
        // No state: every live row, as before (the deleted one stays out).
        val all = eventRepository.pageForAdmin(marker, from, to, "", today, PageRequest.of(0, 50))
        assertEquals(expected.values.flatten().toSet(), all.content.map { it.id }.toSet())
        assertEquals(6, all.totalElements)
        assertEquals(0, eventRepository.countForAdmin(marker, from, to, "nonsense", today))
    }

    @Test
    fun `search matches name and location and dates bound the list`() {
        val marker = "ax9-${UUID.randomUUID()}"
        val place = "ax9place-${UUID.randomUUID()}"
        val byName = save(marker, EventStatus.ACTIVE, today)
        val byPlace = save("other-${UUID.randomUUID()}", EventStatus.ACTIVE, today.minusDays(40), location = "$place, Cebu City")

        assertEquals(setOf(byName), ids(marker.uppercase(), from, to))
        assertEquals(setOf(byPlace), ids(place, from, to))
        assertEquals(setOf(byName), ids(marker, today, today))
        assertEquals(emptySet(), ids(marker, today.plusDays(1), to))
    }

    private fun ids(search: String, dateFrom: LocalDate, dateTo: LocalDate): Set<UUID> =
        eventRepository.pageForAdmin(search, dateFrom, dateTo, "", today, PageRequest.of(0, 50)).content.map { it.id }.toSet()

    private fun save(
        marker: String,
        status: EventStatus,
        date: LocalDate,
        location: String = "Cebu City",
        deletedAt: OffsetDateTime? = null,
    ): UUID =
        eventRepository.save(
            Event(
                slug = "$marker-${UUID.randomUUID()}",
                name = "$marker race",
                date = date,
                location = location,
                status = status,
                deletedAt = deletedAt,
            ),
        ).id
}
