package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoBibEmbed
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.temporal.ChronoUnit
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// AA-R8: the race proof sweep's queries against real Postgres. The container is
// shared, so every bib is unique to its test and dates sit far in the past.
class RaceProofQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `counts and shared bibs match hand-built rows`() {
        val bib = bibPrefix()
        val race = race(DAY)
        val (a, b, c, d) = List(4) { photographer() }
        photo(race, a, IndexingStatus.INDEXED, "${bib}1")
        photo(race, a, IndexingStatus.INDEXED, "${bib}2", "${bib}3")
        photo(race, a, IndexingStatus.INDEXED)
        photo(race, a, IndexingStatus.SKIPPED)
        photo(race, b, IndexingStatus.PENDING)
        photo(race, c, IndexingStatus.INDEXED, "${bib}4", deleted = true)
        // An admin hid it for a flag: it proves nothing.
        photo(race, d, IndexingStatus.INDEXED, "${bib}5", hidden = true)

        val counts = photos.raceProofCounts(race.id)
        assertEquals(2, counts.photographers)
        assertEquals(5, counts.total)
        assertEquals(3, counts.indexed)
        assertEquals(2, counts.withBib)

        val empty = photos.raceProofCounts(race(DAY).id)
        assertEquals(listOf(0L, 0L, 0L, 0L), listOf(empty.photographers, empty.total, empty.indexed, empty.withBib))
    }

    @Test
    fun `thirty shared bibs flag, twenty nine do not, two days apart do not`() {
        val bib = bibPrefix()
        val owner = photographer()
        val thirty = Array(30) { "$bib$it" }
        val race = race(DAY).also { photo(it, owner, IndexingStatus.INDEXED, *thirty) }
        val same = race(DAY, name = "Same Day").also { photo(it, owner, IndexingStatus.INDEXED, *thirty) }
        val next = race(DAY.plusDays(1), name = "Next Day").also {
            photo(it, owner, IndexingStatus.INDEXED, *thirty.copyOf(29).requireNoNulls())
        }
        race(DAY.plusDays(2)).also { photo(it, owner, IndexingStatus.INDEXED, *thirty) }
        race(DAY, status = EventStatus.DRAFT).also { photo(it, owner, IndexingStatus.INDEXED, *thirty) }

        fun shared(min: Int) = photos.eventsSharingBibs(race.id, DAY.minusDays(1), DAY.plusDays(1), min)
            .map { Triple(it.eventId, it.name, it.shared) }

        assertEquals(listOf(Triple(same.id, "Same Day", 30L)), shared(30))
        assertEquals(listOf(Triple(same.id, "Same Day", 30L), Triple(next.id, "Next Day", 29L)), shared(29))
    }

    @Test
    fun `untyped admin events are never selected`() {
        val due = race(DAY)
        val open = race(DAY.plusDays(1))
        val untyped = race(DAY, type = null)
        val draft = race(DAY, status = EventStatus.DRAFT)
        val deleted = race(DAY, deletedAt = OffsetDateTime.now())
        val approved = race(DAY, adminApprovedAt = OffsetDateTime.now())
        val checked = race(DAY).also { tx.execute { _ -> events.claimProofCheck(it.id, now()) } }

        val picked = events.findRacesDueForProof(DAY, PageRequest.of(0, 500)).map { it.id }.toSet()
        assertTrue(due.id in picked && approved.id in picked)
        listOf(open, untyped, draft, deleted, checked).forEach { assertFalse(it.id in picked, it.name) }

        assertEquals(listOf(1, 0), List(2) { tx.execute { _ -> events.claimProofCheck(due.id, now()) } })
        assertEquals(listOf(1, 0), List(2) { tx.execute { _ -> events.hideUnprovenRace(due.id, now()) } })
        assertEquals(0, tx.execute { _ -> events.hideUnprovenRace(approved.id, now()) })
        assertEquals(0, tx.execute { _ -> events.hideUnprovenRace(untyped.id, now()) })
    }

    private fun now() = OffsetDateTime.now().truncatedTo(ChronoUnit.MICROS)

    private fun bibPrefix() = UUID.randomUUID().toString().take(8) + "-"

    private fun photographer(): User = users.save(
        User(email = "rp-${UUID.randomUUID()}@test.local", passwordHash = "stub", name = "RP", role = Role.PHOTOGRAPHER),
    )

    private fun race(
        date: LocalDate,
        name: String = "Proof Run",
        type: RaceType? = RaceType.ROAD_RACE,
        status: EventStatus = EventStatus.ACTIVE,
        deletedAt: OffsetDateTime? = null,
        adminApprovedAt: OffsetDateTime? = null,
    ): Event = events.saveAndFlush(
        Event(
            slug = "rp-${UUID.randomUUID()}",
            name = name,
            date = date,
            location = "Cebu City",
            status = status,
            deletedAt = deletedAt,
            eventType = type,
            adminApprovedAt = adminApprovedAt,
        ),
    )

    private fun photo(
        event: Event, owner: User, status: IndexingStatus, vararg bibs: String, deleted: Boolean = false, hidden: Boolean = false,
    ) {
        photos.saveAndFlush(
            Photo(
                eventId = event.id,
                photographerId = owner.id,
                s3Key = "rp/${UUID.randomUUID()}.jpg",
                pricePhp = BigDecimal("125.00"),
                deletedAt = if (deleted) OffsetDateTime.now() else null,
            ).also { p ->
                p.indexingStatus = status
                if (hidden) p.status = PhotoStatus.HIDDEN
                bibs.forEach { p.bibs.add(PhotoBibEmbed(it, BigDecimal("0.9"))) }
            },
        )
    }

    private companion object {
        val DAY: LocalDate = LocalDate.of(1902, 3, 10)
    }
}
