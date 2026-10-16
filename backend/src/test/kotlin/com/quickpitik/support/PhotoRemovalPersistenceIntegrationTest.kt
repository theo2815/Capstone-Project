package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.removal.*
import com.quickpitik.service.storage.StorageService
import org.flywaydb.core.Flyway
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.dao.OptimisticLockingFailureException
import org.springframework.data.domain.PageRequest
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.IllegalTransactionStateException
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Instant
import java.time.LocalDate
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.sql.DataSource
import kotlin.test.*

class PhotoRemovalPersistenceIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var reviews: PhotoRemovalReviewRepository
    @Autowired lateinit var history: PhotoRemovalHistoryRepository
    @Autowired lateinit var recorder: PhotoRemovalHistoryRecorder
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var dataSource: DataSource
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var provider: AiApiClient
    private val now = Instant.parse("2026-10-05T00:00:00Z")
    private lateinit var owner: User
    private lateinit var event: Event
    private lateinit var photo: Photo
    private lateinit var request: PhotoRemovalRequest
    private lateinit var review: PhotoRemovalReview
    private lateinit var selected: PhotoRemovalItem
    private lateinit var recordedEvent: RemovalHistoryEvent

    @BeforeEach fun fixture() {
        owner = users.saveAndFlush(User(email = "removal-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Actual photographer", role = Role.PHOTOGRAPHER))
        event = events.saveAndFlush(Event(slug = "removal-${UUID.randomUUID()}", name = "Removal run", date = LocalDate.of(2026, 10, 5),
            location = "Cebu", status = EventStatus.ACTIVE))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("125")))
        request = newRequest()
        review = reviews.saveAndFlush(PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = owner.id, ownerId = owner.id))
        selected = items.saveAndFlush(PhotoRemovalItem(requestId = request.id, photoIdSnapshot = photo.id,
            photoId = photo.id, ownerIdSnapshot = owner.id, ownerNameSnapshot = owner.name, reviewId = review.id,
            state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL))
    }

    private fun newRequest() = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id,
        eventNameSnapshot = event.name, requesterContact = "Support-held contact", supportReference = "Support-${UUID.randomUUID()}",
        requesterReason = "Privacy request", publicToOwnerReason = "Verified privacy request", creatorId = UUID.randomUUID(),
        creationCommandId = UUID.randomUUID(), status = RemovalRequestStatus.OPEN))

    private fun audit(commandId: UUID = UUID.randomUUID()) = RemovalHistoryEvent(requestId = request.id, itemId = selected.id,
        reviewId = review.id, photoIdSnapshot = photo.id, actorType = RemovalActorType.ADMIN, actorId = UUID.randomUUID(),
        action = RemovalHistoryActions.ADMIN_DECIDED, fromState = "VERIFIED", toState = "REMOVAL_AUTHORIZED",
        reason = "Verified support request", occurredAt = now, commandId = commandId, metadata = mapOf("source" to "MANUAL"))
    private fun historyFor(commandId: UUID) = history.findByRequestIdAndCommandId(request.id, commandId)
    private data class Failure(val status: Int)
    private fun replayWithDifferentReason(): Failure {
        val error = assertFailsWith<ApiException> { tx.executeWithoutResult { recorder.record(recordedEvent.copy(reason = "Different decision")) } }
        return Failure(GlobalExceptionHandler().handleApi(error).statusCode.value())
    }

    @Test fun `sameCommandCannotChangeMeaning`() {
        val commandId = UUID.randomUUID()
        val original = audit(commandId)
        recordedEvent = original
        tx.executeWithoutResult { recorder.record(original) }
        tx.executeWithoutResult { recorder.record(original) }
        assertEquals(1, historyFor(commandId).size)
        assertEquals(409, replayWithDifferentReason().status)
        for (changed in listOf(original.copy(action = RemovalHistoryActions.ERASURE_PROGRESS),
            original.copy(actorId = UUID.randomUUID()), original.copy(actorType = RemovalActorType.SUPPORT),
            original.copy(photoIdSnapshot = UUID.randomUUID()), original.copy(fromState = "CANDIDATE"),
            original.copy(toState = "RETAINED"), original.copy(occurredAt = now.plusSeconds(1)),
            original.copy(metadata = mapOf("source" to "BIB")))) {
            assertEquals(409, assertFailsWith<ApiException> { tx.executeWithoutResult { recorder.record(changed) } }.status.value())
        }
        assertEquals(1, historyFor(commandId).size)
        assertEquals("Verified support request", historyFor(commandId).single().reason)
    }

    @Test fun `one batch command can record separate items without conflating them`() {
        val other = items.saveAndFlush(PhotoRemovalItem(requestId = request.id, photoIdSnapshot = UUID.randomUUID(),
            state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.BIB))
        val original = audit()
        tx.executeWithoutResult {
            recorder.record(original)
            recorder.record(original.copy(itemId = other.id, reviewId = null, photoIdSnapshot = other.photoIdSnapshot))
        }
        assertEquals(2, historyFor(original.commandId).size)
    }

    @Test fun `nanosecond event time replays identically at database timestamp precision`() {
        val original = audit().copy(occurredAt = Instant.parse("2026-10-05T12:34:56.123456789Z"))
        tx.executeWithoutResult { recorder.record(original) }
        tx.executeWithoutResult { recorder.record(original) }
        assertEquals(1, historyFor(original.commandId).size)
        assertEquals(Instant.parse("2026-10-05T12:34:56.123456Z"), historyFor(original.commandId).single().occurredAt)
        assertEquals(409, assertFailsWith<ApiException> {
            tx.executeWithoutResult { recorder.record(original.copy(occurredAt = original.occurredAt.plusNanos(1000))) }
        }.status.value())
    }

    @Test fun `historySurvivesPhotoAndOwnerRemoval`() {
        val original = audit().copy(actorId = owner.id)
        tx.executeWithoutResult { recorder.record(original) }
        val originalPhotoId = photo.id
        jdbc.update("DELETE FROM photos WHERE id = ?", photo.id)
        jdbc.update("DELETE FROM users WHERE id = ?", owner.id)
        jdbc.update("DELETE FROM events WHERE id = ?", event.id)
        val historyAfterPhotoDelete = historyFor(original.commandId).single()
        assertEquals(originalPhotoId, historyAfterPhotoDelete.photoIdSnapshot)
        assertEquals(owner.id, historyAfterPhotoDelete.actorIdSnapshot)
        val survivingItem = items.findById(selected.id).orElseThrow()
        assertNull(survivingItem.photoId)
        assertEquals(originalPhotoId, survivingItem.photoIdSnapshot)
        assertEquals(owner.id, survivingItem.ownerIdSnapshot)
        assertEquals("Actual photographer", survivingItem.ownerNameSnapshot)
        assertNull(reviews.findById(review.id).orElseThrow().ownerId)
        val survivingRequest = requests.findById(request.id).orElseThrow()
        assertNull(survivingRequest.eventId)
        assertEquals(event.id, survivingRequest.eventIdSnapshot)
        assertEquals("Removal run", survivingRequest.eventNameSnapshot)
        tx.executeWithoutResult { recorder.record(original) }
        assertEquals(1, historyFor(original.commandId).size)
    }

    @Test fun `history requires a transaction and rolls back with its caller`() {
        val original = audit()
        assertFailsWith<IllegalTransactionStateException> { recorder.record(original) }
        tx.executeWithoutResult { status -> recorder.record(original); status.setRollbackOnly() }
        assertTrue(historyFor(original.commandId).isEmpty())
    }

    @Test fun `concurrent command replay appends once and conflicting replay cannot overwrite it`() {
        val original = audit()
        val start = CountDownLatch(1)
        val workers = Executors.newFixedThreadPool(2)
        try {
            val attempts = List(2) { workers.submit {
                start.await()
                tx.executeWithoutResult { recorder.record(original) }
            } }
            start.countDown()
            attempts.forEach { it.get(10, TimeUnit.SECONDS) }
        } finally { workers.shutdownNow() }
        assertEquals(1, historyFor(original.commandId).size)
        assertEquals(409, assertFailsWith<ApiException> {
            tx.executeWithoutResult { recorder.record(original.copy(reason = "Concurrent changed content")) }
        }.status.value())
    }

    @Test fun `optimistic versions prevent a detached case from overwriting a newer update`() {
        val first = requests.findById(request.id).orElseThrow()
        val stale = requests.findById(request.id).orElseThrow()
        first.status = RemovalRequestStatus.RESOLVING
        requests.saveAndFlush(first)
        stale.status = RemovalRequestStatus.CANCELLED
        assertFailsWith<OptimisticLockingFailureException> { requests.saveAndFlush(stale) }
        val fresh = requests.findById(request.id).orElseThrow()
        assertEquals(RemovalRequestStatus.RESOLVING, fresh.status)
        assertEquals(request.version + 1, fresh.version)
    }

    @Test fun `foreign request item and review membership cannot enter audit`() {
        val foreignRequest = newRequest()
        val foreignReview = reviews.saveAndFlush(PhotoRemovalReview(requestId = foreignRequest.id, ownerIdSnapshot = owner.id))
        val foreignItem = items.saveAndFlush(PhotoRemovalItem(requestId = foreignRequest.id, photoIdSnapshot = UUID.randomUUID(), matchSource = RemovalMatchSource.MANUAL))
        for (bad in listOf(audit().copy(itemId = foreignItem.id), audit().copy(reviewId = foreignReview.id),
            audit().copy(requestId = UUID.randomUUID()), audit().copy(photoIdSnapshot = UUID.randomUUID()))) {
            val error = assertFailsWith<ApiException> { tx.executeWithoutResult { recorder.record(bad) } }
            assertTrue(error.status.value() in setOf(404, 409))
            assertTrue(historyFor(bad.commandId).isEmpty())
        }
        assertNull(items.findByIdAndRequestId(selected.id, foreignRequest.id))
        assertNull(reviews.findByIdAndRequestId(review.id, foreignRequest.id))
        assertEquals(listOf(selected.id), items.findByRequestIdOrderByIdAsc(request.id).map { it.id })
        assertEquals(listOf(review.id), reviews.findByRequestIdOrderByIdAsc(request.id).map { it.id })
        assertEquals(request.id, requests.findByCreationCommandId(request.creationCommandId)?.id)
    }

    @Test fun `history actions reasons and metadata reject unsafe content`() {
        for (bad in listOf(audit().copy(action = "CLIENT_STRING"), audit().copy(reason = " "),
            audit().copy(metadata = mapOf("selfieBytes" to "payload")),
            audit().copy(metadata = mapOf("embedding" to "0.1,0.2")),
            audit().copy(metadata = mapOf("signedUrl" to "https://storage.test/image?signature=secret")))) {
            assertEquals(400, assertFailsWith<ApiException> { tx.executeWithoutResult { recorder.record(bad) } }.status.value())
            assertTrue(historyFor(bad.commandId).isEmpty())
        }
    }

    @Test fun `database protects unique creation photo and owner records`() {
        assertFailsWith<DataIntegrityViolationException> {
            requests.saveAndFlush(PhotoRemovalRequest(eventIdSnapshot = event.id, eventNameSnapshot = event.name,
                requesterContact = "Contact", supportReference = "Support", requesterReason = "Reason", publicToOwnerReason = "Reason",
                creatorId = UUID.randomUUID(), creationCommandId = request.creationCommandId))
        }
        assertFailsWith<DataIntegrityViolationException> {
            items.saveAndFlush(PhotoRemovalItem(requestId = request.id, photoIdSnapshot = photo.id, matchSource = RemovalMatchSource.MANUAL))
        }
        assertFailsWith<DataIntegrityViolationException> {
            reviews.saveAndFlush(PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = owner.id))
        }
    }

    @Test fun `database rejects cross request reviews and history mutation`() {
        val foreignReview = reviews.saveAndFlush(PhotoRemovalReview(requestId = newRequest().id, ownerIdSnapshot = owner.id))
        assertFailsWith<DataIntegrityViolationException> { jdbc.update("UPDATE photo_removal_items SET review_id = ? WHERE id = ?", foreignReview.id, selected.id) }
        val original = audit()
        tx.executeWithoutResult { recorder.record(original) }
        val row = historyFor(original.commandId).single()
        assertFailsWith<DataIntegrityViolationException> { jdbc.update("UPDATE photo_removal_history SET reason = 'changed' WHERE id = ?", row.id) }
        assertFailsWith<DataIntegrityViolationException> { jdbc.update("DELETE FROM photo_removal_history WHERE id = ?", row.id) }
        assertEquals("Verified support request", historyFor(original.commandId).single().reason)
    }

    @Test fun `database freezes event photo and owner snapshots`() {
        for ((sql, id) in listOf(
            "UPDATE photo_removal_requests SET event_id_snapshot = ? WHERE id = ?" to request.id,
            "UPDATE photo_removal_items SET photo_id_snapshot = ? WHERE id = ?" to selected.id,
            "UPDATE photo_removal_items SET owner_id_snapshot = ? WHERE id = ?" to selected.id,
            "UPDATE photo_removal_reviews SET owner_id_snapshot = ? WHERE id = ?" to review.id,
        )) assertFailsWith<DataIntegrityViolationException> { jdbc.update(sql, UUID.randomUUID(), id) }
        assertFailsWith<DataIntegrityViolationException> { jdbc.update("UPDATE photo_removal_items SET owner_name_snapshot = 'changed' WHERE id = ?", selected.id) }
        assertFailsWith<DataIntegrityViolationException> { jdbc.update("UPDATE photo_removal_requests SET event_name_snapshot = 'changed' WHERE id = ?", request.id) }
    }

    @Test fun `database rejects unknown state and blank intake or decision reasons`() {
        for ((sql, id) in listOf(
            "UPDATE photo_removal_requests SET status = 'UNKNOWN' WHERE id = ?" to request.id,
            "UPDATE photo_removal_requests SET requester_reason = ' ' WHERE id = ?" to request.id,
            "UPDATE photo_removal_requests SET support_reference = ' ' WHERE id = ?" to request.id,
            "UPDATE photo_removal_items SET state = 'UNKNOWN' WHERE id = ?" to selected.id,
            "UPDATE photo_removal_items SET admin_decision = 'REMOVE', admin_decision_reason = ' ' WHERE id = ?" to selected.id,
        )) assertFailsWith<DataIntegrityViolationException> { jdbc.update(sql, id) }
    }

    @Test fun `due reviews page only notified active cases once in deadline order`() {
        review.notifiedAt = now.minusSeconds(72 * 3600)
        review.deadline = now.minusSeconds(2)
        reviews.saveAndFlush(review)
        val laterRequest = newRequest()
        val later = reviews.saveAndFlush(PhotoRemovalReview(requestId = laterRequest.id, ownerIdSnapshot = owner.id,
            notifiedAt = now.minusSeconds(72 * 3600), deadline = now.minusSeconds(1)))
        for ((status, deadline, notified, overdue, unavailable) in listOf(
            listOf(RemovalRequestStatus.OPEN, now.plusSeconds(1), now, false, null),
            listOf(RemovalRequestStatus.RESOLVED, now.minusSeconds(1), now, false, null),
            listOf(RemovalRequestStatus.DRAFT, now.minusSeconds(1), now, false, null),
            listOf(RemovalRequestStatus.CANCELLED, now.minusSeconds(1), now, false, null),
            listOf(RemovalRequestStatus.OPEN, now.minusSeconds(1), null, false, null),
            listOf(RemovalRequestStatus.OPEN, now.minusSeconds(1), now, true, null),
            listOf(RemovalRequestStatus.OPEN, now.minusSeconds(1), now, false, "Unreachable"),
        )) {
            val excluded = newRequest().also { it.status = status as RemovalRequestStatus }
            requests.saveAndFlush(excluded)
            reviews.saveAndFlush(PhotoRemovalReview(requestId = excluded.id, ownerIdSnapshot = owner.id,
                notifiedAt = notified as Instant?, deadline = deadline as Instant, overdue = overdue as Boolean,
                unavailableReason = unavailable as String?))
        }
        assertEquals(listOf(review.id), reviews.findDueReviews(now, PageRequest.of(0, 1)).map { it.id })
        assertEquals(listOf(later.id), reviews.findDueReviews(now, PageRequest.of(1, 1)).map { it.id })
        assertTrue(reviews.findDueReviews(now, PageRequest.of(2, 1)).isEmpty())
    }

    @Test fun `history pages stay inside a request and keep stable ordering`() {
        val original = audit()
        tx.executeWithoutResult {
            recorder.record(original)
            recorder.record(original.copy(commandId = UUID.randomUUID(), occurredAt = now.plusSeconds(1)))
            recorder.record(original.copy(requestId = newRequest().id, itemId = null, reviewId = null, photoIdSnapshot = null,
                commandId = UUID.randomUUID(), occurredAt = now.minusSeconds(1)))
        }
        val first = history.findByRequestIdOrderByOccurredAtAscIdAsc(request.id, PageRequest.of(0, 1))
        val second = history.findByRequestIdOrderByOccurredAtAscIdAsc(request.id, PageRequest.of(1, 1))
        assertEquals(now, first.single().occurredAt)
        assertEquals(now.plusSeconds(1), second.single().occurredAt)
        assertTrue(history.findByRequestIdOrderByOccurredAtAscIdAsc(request.id, PageRequest.of(2, 1)).isEmpty())
    }

    @Test fun `V68 migrates empty databases and upgrades existing V67 photos without backfill`() {
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM flyway_schema_history WHERE version = '68' AND success", Int::class.java))
        val schema = "removal_upgrade_${UUID.randomUUID().toString().replace("-", "")}"
        fun migrate(target: String) = Flyway.configure().dataSource(dataSource).locations("classpath:db/migration")
            .defaultSchema(schema).schemas(schema).target(target).load().migrate()
        val oldPhoto = UUID.randomUUID()
        try {
            migrate("67")
            jdbc.update("INSERT INTO $schema.photos (id, event_id, s3_key, status, price_php) VALUES (?, ?, 'synthetic/legacy', 'LIVE', 125)",
                oldPhoto, UUID.fromString("11111111-1111-1111-1111-111111111111"))
            migrate("68")
            assertEquals("synthetic/legacy", jdbc.queryForObject("SELECT s3_key FROM $schema.photos WHERE id = ?", String::class.java, oldPhoto))
            for (table in listOf("requests", "items", "reviews", "history")) {
                assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM $schema.photo_removal_$table", Int::class.java))
            }
        } finally { jdbc.execute("DROP SCHEMA IF EXISTS $schema CASCADE") }
    }
}
