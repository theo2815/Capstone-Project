package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventPhotoAlertRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class EventPhotoAlertRepositoryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var alertRepository: EventPhotoAlertRepository

    @Autowired
    private lateinit var eventRepository: EventRepository

    @Autowired
    private lateinit var userRepository: UserRepository

    @Test
    fun `pending sweep covers completed grace-day events and rotates checked alerts`() {
        val today = LocalDate.now()
        val active = event(EventStatus.ACTIVE, today)
        val completedGraceDay = event(EventStatus.COMPLETED, today.minusDays(4))
        val archived = event(EventStatus.ARCHIVED, today)
        val expired = event(EventStatus.ACTIVE, today.minusDays(5))

        val neverChecked = alert(active.id)
        val alreadyChecked = alert(active.id, OffsetDateTime.now().minusHours(1))
        val completedAlert = alert(completedGraceDay.id)
        val archivedAlert = alert(archived.id)
        val expiredAlert = alert(expired.id)

        val pending = alertRepository.findPendingInWindow(
            today,
            today.minusDays(4),
            PageRequest.of(0, 100),
        ).map { it.id }

        assertContains(pending, neverChecked.id)
        assertContains(pending, completedAlert.id)
        assertFalse(archivedAlert.id in pending)
        assertFalse(expiredAlert.id in pending)
        assertTrue(pending.indexOf(neverChecked.id) < pending.indexOf(alreadyChecked.id))
    }

    private fun event(status: EventStatus, date: LocalDate): Event = eventRepository.save(
        Event(
            slug = "photo-alert-${UUID.randomUUID()}",
            name = "Photo Alert Test",
            date = date,
            location = "Cebu City",
            status = status,
        ),
    )

    private fun alert(eventId: UUID, lastCheckedAt: OffsetDateTime? = null): EventPhotoAlert =
        alertRepository.save(
            EventPhotoAlert(
                eventId = eventId,
                userId = userRepository.save(
                    User(
                        email = "runner-${UUID.randomUUID()}@test.local",
                        passwordHash = "stub",
                        name = "Test Runner",
                        role = Role.RUNNER,
                    ),
                ).id,
                lastCheckedAt = lastCheckedAt,
            ),
        )
}
