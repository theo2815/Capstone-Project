package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.UpdateRaceRequest
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.events.DuplicateMatcher
import com.quickpitik.service.events.RaceEventService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.TestPropertySource
import org.springframework.transaction.support.TransactionTemplate
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// AA-R5: the race edit's two queries against real Postgres; AA-R10: the edit path meets the unique index.
@TestPropertySource(properties = ["app.races.self-serve-enabled=true"])
class RaceEditQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var races: RaceEventService

    // A pending race takes an edit at once, so a rename onto a live race's date and key meets the
    // unique index at flush; the service answers 409, never a 500 (review F32).
    @Test
    fun `a submitter rename onto another race's date and key answers RACE_DUPLICATE`() {
        val title = "Rename Clash Marathon ${UUID.randomUUID().toString().filter(Char::isLetter).take(6)}"
        val date = LocalDate.now().plusDays(30)
        race(photographer(), name = title, date = date, key = DuplicateMatcher(0.0, 0.0, 0).nameKey(title))
        val me = photographer()
        jdbc.update("INSERT INTO photographer_settings (user_id, verification_status) VALUES (?, 'APPROVED')", me.id)
        val mine = race(me, review = EventReviewStatus.PENDING, status = EventStatus.DRAFT, date = date)

        val error = assertFailsWith<ConflictException> { races.update(me.id, mine.id, UpdateRaceRequest(title = title), null) }
        assertEquals(ErrorCodes.RACE_DUPLICATE, error.code)
    }

    @Test
    fun `a rejected race is no duplicate candidate`() {
        val me = photographer()
        val live = race(me)
        val rejected = race(me, review = EventReviewStatus.REJECTED)
        val date = live.date
        val candidates = events.findRaceCandidates(date, date).map { it.id }
        assertTrue(live.id in candidates)
        assertFalse(rejected.id in candidates)
    }

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
        name: String = "Edit Run ${review.name}",
        date: LocalDate = LocalDate.of(1903, 4, 12),
        key: String? = null,
    ): Event = events.saveAndFlush(
        Event(
            slug = "re-${UUID.randomUUID()}",
            name = name,
            date = date,
            location = "Cebu City",
            status = status,
            deletedAt = deletedAt,
            eventType = RaceType.ROAD_RACE,
            reviewStatus = review,
        ).apply { submittedBy = submitter.id; nameKey = key },
    )
}
