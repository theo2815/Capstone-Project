package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.transaction.support.TransactionTemplate
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// AA-R5: the race edit's two queries against real Postgres.
class RaceEditQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `only the submitter's live or pending typed race is found for an edit`() {
        val me = photographer()
        val other = photographer()
        val race = race(me)
        val pending = race(me, review = EventReviewStatus.PENDING, status = EventStatus.DRAFT)
        // No untyped case: ck_events_race_only refuses submitted_by on an untyped row.
        val rejected = race(me, review = EventReviewStatus.REJECTED)
        val deleted = race(me, deletedAt = OffsetDateTime.now())

        fun find(id: UUID, by: UUID) = tx.execute { _ -> events.findRaceForSubmitterForUpdate(id, by) }
        assertEquals(race.id, find(race.id, me.id)?.id)
        assertEquals(pending.id, find(pending.id, me.id)?.id)
        assertNull(find(race.id, other.id))
        listOf(rejected, deleted).forEach { assertNull(find(it.id, me.id), it.name) }
    }

    @Test
    fun `another photographer's first upload locks the edit, a bare join does not`() {
        val me = photographer()
        val joiner = photographer()
        val uploader = photographer()
        val race = race(me)
        coverage.saveAndFlush(EventPhotographer(id = EventPhotographerId(race.id, me.id), firstUploadAt = OffsetDateTime.now()))
        coverage.saveAndFlush(EventPhotographer(id = EventPhotographerId(race.id, joiner.id)))

        assertFalse(coverage.existsOtherUploader(race.id, me.id))

        coverage.saveAndFlush(EventPhotographer(id = EventPhotographerId(race.id, uploader.id), firstUploadAt = OffsetDateTime.now()))
        assertTrue(coverage.existsOtherUploader(race.id, me.id))
        assertTrue(coverage.existsOtherUploader(race.id, joiner.id))
    }

    private fun photographer(): User = users.save(
        User(email = "re-${UUID.randomUUID()}@test.local", passwordHash = "stub", name = "RE", role = Role.PHOTOGRAPHER),
    )

    private fun race(
        submitter: User,
        review: EventReviewStatus = EventReviewStatus.APPROVED,
        status: EventStatus = EventStatus.ACTIVE,
        deletedAt: OffsetDateTime? = null,
    ): Event = events.saveAndFlush(
        Event(
            slug = "re-${UUID.randomUUID()}",
            name = "Edit Run ${review.name}",
            date = LocalDate.of(1903, 4, 12),
            location = "Cebu City",
            status = status,
            deletedAt = deletedAt,
            eventType = RaceType.ROAD_RACE,
            reviewStatus = review,
        ).apply { submittedBy = submitter.id },
    )
}
