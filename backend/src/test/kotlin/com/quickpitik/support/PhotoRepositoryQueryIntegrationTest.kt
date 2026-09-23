package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.*

// The PhotoRepository @Query methods no other Postgres test runs (P2-5 / P-16).
// Mocks never parse JPQL, so a wrong predicate or enum binding only shows here.
// The database is shared across classes: every assertion is scoped to this
// test's own event, and global sweeps are read with a page far above their size.
class PhotoRepositoryQueryIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var tx: TransactionTemplate
    lateinit var event: Event
    lateinit var otherEvent: Event
    lateinit var owner: User

    private val statuses = listOf(IndexingStatus.PENDING, IndexingStatus.FAILED)
    private val old = OffsetDateTime.now().minusHours(1)
    private val cutoff = OffsetDateTime.now().minusMinutes(5)
    private val everything = PageRequest.of(0, 10_000)

    @BeforeEach fun seed() {
        owner = users.save(User(email = "repo-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Repo", role = Role.PHOTOGRAPHER))
        event = events.save(newEvent())
        otherEvent = events.save(newEvent())
    }

    @Test fun `resetForRewatermark sends only the event's LIVE photos back to PROCESSING with a fresh budget`() {
        val live = save(status = PhotoStatus.LIVE, processingAttempts = 2)
        val hidden = save(status = PhotoStatus.HIDDEN)
        val foreign = save(eventId = otherEvent.id, status = PhotoStatus.LIVE)

        assertEquals(1, inTx { photos.resetForRewatermark(event.id) })
        val reset = reload(live)
        assertEquals(PhotoStatus.PROCESSING, reset.status)
        assertEquals(0, reset.processingAttempts)
        assertEquals(PhotoStatus.HIDDEN, reload(hidden).status)
        assertEquals(PhotoStatus.LIVE, reload(foreign).status)
    }

    @Test fun `updatePriceByEventId reprices every photo in the event and nothing else`() {
        val a = save(status = PhotoStatus.LIVE)
        val b = save(status = PhotoStatus.HIDDEN)
        val foreign = save(eventId = otherEvent.id)

        assertEquals(2, inTx { photos.updatePriceByEventId(event.id, BigDecimal("80.00")) })
        assertEquals(0, BigDecimal("80").compareTo(reload(a).pricePhp))
        assertEquals(0, BigDecimal("80").compareTo(reload(b).pricePhp))
        assertEquals(0, BigDecimal("125").compareTo(reload(foreign).pricePhp))
    }

    @Test fun `requeueIndexing resets the listed statuses to PENDING with a fresh budget`() {
        val failed = save(indexingStatus = IndexingStatus.FAILED, indexingAttempts = 5, indexingError = "boom")
        val indexed = save(indexingStatus = IndexingStatus.INDEXED, indexingAttempts = 1)
        val foreign = save(eventId = otherEvent.id, indexingStatus = IndexingStatus.FAILED)

        assertEquals(1, inTx { photos.requeueIndexing(event.id, listOf(IndexingStatus.FAILED)) })
        val requeued = reload(failed)
        assertEquals(IndexingStatus.PENDING, requeued.indexingStatus)
        assertEquals(0, requeued.indexingAttempts)
        assertNull(requeued.indexingError)
        assertEquals(IndexingStatus.INDEXED, reload(indexed).indexingStatus)
        assertEquals(IndexingStatus.FAILED, reload(foreign).indexingStatus)
    }

    @Test fun `incrementProcessingAttempts adds one per call`() {
        val photo = save(processingAttempts = 1)
        assertEquals(1, inTx { photos.incrementProcessingAttempts(photo.id) })
        assertEquals(1, inTx { photos.incrementProcessingAttempts(photo.id) })
        assertEquals(3, reload(photo).processingAttempts)
    }

    @Test fun `findPhotographerLibrary returns the owner's untrashed photos in the event at any status`() {
        val live = save(status = PhotoStatus.LIVE)
        val processing = save(status = PhotoStatus.PROCESSING)
        save(deletedAt = OffsetDateTime.now())
        save(photographerId = null)
        save(eventId = otherEvent.id)

        val page = photos.findPhotographerLibrary(event.id, owner.id, PageRequest.of(0, 20))
        assertEquals(setOf(live.id, processing.id), page.content.map { it.id }.toSet())
        assertEquals(2L, page.totalElements)
    }

    @Test fun `the indexing backlog holds settled-late, under-budget, untrashed photos of the listed statuses`() {
        val eligible = save(indexingStatus = IndexingStatus.FAILED, uploadedAt = old)
        val recent = save(indexingStatus = IndexingStatus.PENDING, uploadedAt = OffsetDateTime.now())
        val exhausted = save(indexingStatus = IndexingStatus.PENDING, uploadedAt = old, indexingAttempts = 3)
        val trashed = save(indexingStatus = IndexingStatus.PENDING, uploadedAt = old, deletedAt = OffsetDateTime.now())
        val indexed = save(indexingStatus = IndexingStatus.INDEXED, uploadedAt = old)
        val mine = setOf(eligible.id, recent.id, exhausted.id, trashed.id, indexed.id)

        val backlog = photos.findIndexingBacklog(statuses, 3, cutoff, everything).map { it.id }.filter { it in mine }
        assertEquals(listOf(eligible.id), backlog)
    }

    @Test fun `findEventsWithIndexingBacklog lists an event only when it has an eligible photo`() {
        save(indexingStatus = IndexingStatus.PENDING, uploadedAt = old)
        save(indexingStatus = IndexingStatus.FAILED, uploadedAt = old)
        save(eventId = otherEvent.id, indexingStatus = IndexingStatus.PENDING, uploadedAt = OffsetDateTime.now())

        val eventIds = photos.findEventsWithIndexingBacklog(statuses, 3, cutoff, everything)
        assertTrue(event.id in eventIds)
        assertFalse(otherEvent.id in eventIds)
        assertEquals(1, eventIds.count { it == event.id }, "DISTINCT must collapse the event to one row")
    }

    @Test fun `findIndexingBacklogForEvent is scoped to the event and oldest first`() {
        val newer = save(indexingStatus = IndexingStatus.PENDING, uploadedAt = old)
        val oldest = save(indexingStatus = IndexingStatus.FAILED, uploadedAt = old.minusHours(1))
        save(indexingStatus = IndexingStatus.PENDING, uploadedAt = old, indexingAttempts = 3)
        save(eventId = otherEvent.id, indexingStatus = IndexingStatus.PENDING, uploadedAt = old)

        val backlog = photos.findIndexingBacklogForEvent(event.id, statuses, 3, cutoff, PageRequest.of(0, 20)).map { it.id }
        assertEquals(listOf(oldest.id, newer.id), backlog)
    }

    @Test fun `findReferencedAiPersonIds covers every status in the event and no other event`() {
        save(status = PhotoStatus.LIVE, personIds = listOf("p-live", "p-shared"))
        save(status = PhotoStatus.HIDDEN, personIds = listOf("p-hidden", "p-shared"))
        save(eventId = otherEvent.id, personIds = listOf("p-foreign"))

        val ids = photos.findReferencedAiPersonIds(event.id)
        assertEquals(setOf("p-live", "p-hidden", "p-shared"), ids.toSet())
        assertEquals(3, ids.size, "DISTINCT must collapse a person shared by two photos")
    }

    @Test fun `findEventsWithFacePersons lists only events that hold an indexed face`() {
        save(personIds = listOf("p-${UUID.randomUUID()}"))
        save(eventId = otherEvent.id)

        val eventIds = photos.findEventsWithFacePersons(everything)
        assertTrue(event.id in eventIds)
        assertFalse(otherEvent.id in eventIds)
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun inTx(block: () -> Int): Int = tx.execute { block() }!!

    private fun reload(photo: Photo) = photos.findById(photo.id).orElseThrow()

    private fun newEvent() = Event(slug = "repo-${UUID.randomUUID()}", name = "Repo", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE)

    private fun save(
        eventId: UUID = event.id,
        photographerId: UUID? = owner.id,
        status: PhotoStatus = PhotoStatus.LIVE,
        processingAttempts: Int = 0,
        indexingStatus: IndexingStatus = IndexingStatus.INDEXED,
        indexingAttempts: Int = 0,
        indexingError: String? = null,
        uploadedAt: OffsetDateTime = OffsetDateTime.now(),
        deletedAt: OffsetDateTime? = null,
        personIds: List<String> = emptyList(),
    ): Photo = photos.saveAndFlush(
        Photo(
            eventId = eventId,
            photographerId = photographerId,
            s3Key = "repo/${UUID.randomUUID()}.jpg",
            pricePhp = BigDecimal("125"),
            status = status,
            processingAttempts = processingAttempts,
            indexingStatus = indexingStatus,
            indexingAttempts = indexingAttempts,
            indexingError = indexingError,
            uploadedAt = uploadedAt,
            deletedAt = deletedAt,
        ).apply { personIds.forEachIndexed { i, id -> facePersons.add(PhotoFacePersonEmbed(faceIndex = i, aiPersonId = id)) } },
    )
}
