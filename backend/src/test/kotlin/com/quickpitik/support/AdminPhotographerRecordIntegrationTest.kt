package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminPhotographerRecordService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

// AX-13: the admin record lists every event the photographer is on, a draft and
// a private one included, newest first; deleted events and other people's are left out.
class AdminPhotographerRecordIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var service: AdminPhotographerRecordService
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var users: UserRepository

    @Test
    fun `the record lists drafts and private events, newest first, with totals`() {
        val photographer = photographer()
        val other = photographer()
        val draft = event(EventStatus.DRAFT, EventVisibility.PUBLIC, LocalDate.of(2026, 12, 6))
        val private = event(EventStatus.ACTIVE, EventVisibility.UNLISTED, LocalDate.of(2026, 10, 4))
        val deleted = event(EventStatus.ACTIVE, EventVisibility.PUBLIC, LocalDate.of(2026, 11, 1), OffsetDateTime.now())
        val othersEvent = event(EventStatus.ACTIVE, EventVisibility.PUBLIC, LocalDate.of(2026, 11, 8))
        cover(draft, photographer, photos = 0, sales = 0)
        cover(private, photographer, photos = 30, sales = 4)
        cover(deleted, photographer, photos = 9, sales = 1)
        cover(othersEvent, other, photos = 50, sales = 5)

        val record = service.record(photographer.id)

        assertEquals(listOf(draft.id, private.id), record.events.map { it.eventId })
        assertEquals(listOf("public", "unlisted"), record.events.map { it.visibility })
        assertEquals(listOf("2026-12-06", "2026-10-04"), record.events.map { it.date })
        assertEquals(listOf(0, 30), record.events.map { it.photoCount })
        assertEquals(2, record.eventsTotal)
        assertEquals(4, record.salesCount)
        assertEquals(30, record.photoCount)
    }

    private fun photographer(): User = users.saveAndFlush(
        User(
            email = "admin-record-${UUID.randomUUID()}@test.local",
            passwordHash = "unused",
            name = "Admin Record",
            role = Role.PHOTOGRAPHER,
        ),
    )

    private fun event(
        status: EventStatus,
        visibility: EventVisibility,
        date: LocalDate,
        deletedAt: OffsetDateTime? = null,
    ): Event = events.saveAndFlush(
        Event(
            slug = "admin-record-${UUID.randomUUID()}",
            name = "Admin Record",
            date = date,
            location = "Cebu City",
            status = status,
            visibility = visibility,
            deletedAt = deletedAt,
        ),
    )

    private fun cover(event: Event, photographer: User, photos: Int, sales: Int) {
        coverage.saveAndFlush(
            EventPhotographer(
                id = EventPhotographerId(event.id, photographer.id),
                photoCount = photos,
                salesCount = sales,
            ),
        )
    }
}
