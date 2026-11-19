package com.quickpitik.service.admin

import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.AdminRecordEventRow
import com.quickpitik.repository.AdminRecordTotalsRow
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageRequest
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

// AX-13: the admin photographer record, whatever the photographer's status.
class AdminPhotographerRecordServiceTest {
    private val users = Mockito.mock(UserRepository::class.java)
    private val coverage = Mockito.mock(EventPhotographerRepository::class.java)
    private val sellerDetails = Mockito.mock(SellerDetailsRepository::class.java)
    private val service = AdminPhotographerRecordService(users, coverage, sellerDetails)
    private val userId = UUID.randomUUID()

    @Test
    fun `an unknown user is a 404`() {
        Mockito.`when`(users.existsById(userId)).thenReturn(false)

        assertFailsWith<NotFoundException> { service.record(userId) }
    }

    @Test
    fun `the record carries the events, totals and seller-details state`() {
        val eventId = UUID.randomUUID()
        Mockito.`when`(users.existsById(userId)).thenReturn(true)
        Mockito.`when`(coverage.findAdminRecordEvents(userId, PageRequest.of(0, 200)))
            .thenReturn(listOf(row(eventId, "Cebu Marathon", LocalDate.of(2026, 11, 1), EventVisibility.UNLISTED, 42)))
        Mockito.`when`(coverage.adminRecordTotals(userId)).thenReturn(totals(events = 3, sales = 7, photos = 120))
        val details = Mockito.mock(SellerDetails::class.java)
        Mockito.`when`(details.status).thenReturn(SellerDetailsStatus.MANUAL_REVIEW)
        Mockito.`when`(sellerDetails.findById(userId)).thenReturn(Optional.of(details))

        val record = service.record(userId)

        val event = record.events.single()
        assertEquals(eventId, event.eventId)
        assertEquals("Cebu Marathon", event.name)
        assertEquals("2026-11-01", event.date)
        assertEquals("unlisted", event.visibility)
        assertEquals(42, event.photoCount)
        assertEquals(3, record.eventsTotal)
        assertEquals(7, record.salesCount)
        assertEquals(120, record.photoCount)
        assertEquals("manual_review", record.sellerDetailsStatus)
    }

    @Test
    fun `no seller details means a null status`() {
        Mockito.`when`(users.existsById(userId)).thenReturn(true)
        Mockito.`when`(coverage.findAdminRecordEvents(userId, PageRequest.of(0, 200))).thenReturn(emptyList())
        Mockito.`when`(coverage.adminRecordTotals(userId)).thenReturn(totals(events = 0, sales = 0, photos = 0))
        Mockito.`when`(sellerDetails.findById(userId)).thenReturn(Optional.empty())

        val record = service.record(userId)

        assertNull(record.sellerDetailsStatus)
        assertEquals(0, record.eventsTotal)
    }

    private fun row(id: UUID, name: String, date: LocalDate, visibility: EventVisibility, photos: Int) =
        object : AdminRecordEventRow {
            override val eventId = id
            override val name = name
            override val date = date
            override val visibility = visibility
            override val photoCount = photos
        }

    private fun totals(events: Long, sales: Long, photos: Long) = object : AdminRecordTotalsRow {
        override val eventsTotal = events
        override val salesCount = sales
        override val photoCount = photos
    }
}
