package com.quickpitik.support

import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.photographer.PhotographerMessageService
import com.quickpitik.service.removal.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Import
import org.springframework.context.annotation.Primary
import org.springframework.data.domain.PageRequest
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.*

@SpringBootTest(properties = ["app.photo-removal.cleanup-interval=PT24H"])
@Import(PhotoRemovalVerificationIntegrationTest.ClockConfig::class)
class PhotoRemovalVerificationIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var service: PhotoRemovalService
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var notifier: PhotoRemovalNotifier
    @Autowired lateinit var inbox: PhotographerMessageService
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var commands: PhotoRemovalCommandRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @MockitoSpyBean lateinit var reviews: PhotoRemovalReviewRepository
    @Autowired lateinit var messages: PhotographerMessageRepository
    @MockitoSpyBean lateinit var recorder: PhotoRemovalHistoryRecorder
    @Autowired lateinit var history: PhotoRemovalHistoryRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var entityManager: jakarta.persistence.EntityManager
    @Autowired lateinit var clock: MutableClock
    private val now = Instant.parse("2026-10-05T00:00:00Z")
    private lateinit var admin: User
    private lateinit var event: Event

    @BeforeEach fun fixture() {
        clock.set(now)
        admin = user(Role.ADMIN)
        event = events.saveAndFlush(Event(slug = "verify-${UUID.randomUUID()}", name = "Synthetic run",
            date = LocalDate.of(2026, 10, 5), location = "Cebu", status = EventStatus.ACTIVE))
    }

    // A creator fallback, wrong group, false deadline or lost restriction breaks this test.
    @Test fun `oneCaseWithThreeOwnersCreatesThreeNotices`() {
        val ownerA = user(); val ownerB = user(); val ownerC = user()
        val selected = listOf(photo(ownerA), photo(ownerA), photo(ownerB), photo(ownerC))
        val case = create()
        val receipt = service.verify(admin.id, case.id, command(case, selected))
        assertTrue(receipt.reviews.all { it.notifiedAt == null }, "original receipt records pending notification")
        val storedMessages = reviews.findByRequestIdOrderByIdAsc(case.id).map { messages.findById(it.notifiedMessageId!!).orElseThrow() }
        assertEquals(setOf(ownerA.id, ownerB.id, ownerC.id), storedMessages.map { it.photographerId }.toSet())
        assertEquals(3, storedMessages.size)
        val reviewCreatedAtOct5 = reviews.findByRequestIdOrderByIdAsc(case.id).first()
        assertEquals(Instant.parse("2026-10-08T00:00:00Z"), reviewCreatedAtOct5.deadline)
        val ownerAItemIds = items.findByRequestIdOrderByIdAsc(case.id).filter { it.ownerIdSnapshot == ownerA.id }.map { it.id }.toSet()
        val noticeForA = storedMessages.single { it.photographerId == ownerA.id }
        val noticeItemIds = UUID_REGEX.findAll(noticeForA.body).map { UUID.fromString(it.value) }.toList()
        assertTrue(noticeItemIds.isNotEmpty())
        assertTrue(noticeItemIds.all { it in ownerAItemIds })
        for (message in storedMessages) {
            assertEquals("photo_removal_request", message.kindWire)
            assertEquals(case.id, message.photoRemovalRequestId)
            assertEquals(now, message.createdAt.toInstant())
            assertFalse(message.body.contains("private contact"))
            assertFalse(message.body.contains("private reason"))
            assertFalse(message.body.contains("private verification"))
        }
        assertTrue(selected.all { photos.findById(it.id).orElseThrow().removalState == RemovalState.REVIEW })
        val detail = queries.adminDetail(admin.id, case.id)
        assertEquals(RemovalRequestStatus.OPEN, detail.status)
        assertTrue(detail.items.all { it.verificationActorId == admin.id && it.verifiedAt == now })
        assertEquals(4, history.findByRequestIdAndCommandId(case.id, receiptCommandId!!).count { it.action == "VERIFIED" && it.itemId != null })
    }

    @Test fun `orphanPhotoNeverUsesEventCreator`() {
        val creator = user()
        event.createdBy = creator.id
        events.saveAndFlush(event)
        val orphan = photo(null)
        val case = create()
        service.verify(admin.id, case.id, command(case, listOf(orphan)))
        val orphanItem = items.findByRequestIdOrderByIdAsc(case.id).single()
        assertNull(orphanItem.reviewId)
        assertNull(orphanItem.ownerIdSnapshot)
        assertTrue(reviews.findByRequestIdOrderByIdAsc(case.id).isEmpty())
        assertTrue(messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(creator.id).isEmpty())
        assertTrue(history.findByRequestIdOrderByOccurredAtAscIdAsc(case.id, PageRequest.of(0, 100)).any {
            it.action == "OWNER_UNAVAILABLE" && it.itemId == orphanItem.id && it.reason != null
        })
    }

    @Test fun `verificationReplayCannotDoubleNotify`() {
        val owners = listOf(user(), user(), user())
        val case = create()
        val command = command(case, owners.map(::photo))
        val original = service.verify(admin.id, case.id, command)
        val deadlines = reviews.findByRequestIdOrderByIdAsc(case.id).associate { it.id to it.deadline }
        clock.set(now.plusSeconds(3600))
        assertEquals(original, service.verify(admin.id, case.id, command))
        reviews.findByRequestIdOrderByIdAsc(case.id).forEach { notifier.notifyReview(it.id) }
        val messagesAfterIdenticalReplay = owners.flatMap { messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(it.id) }
        assertEquals(3, messagesAfterIdenticalReplay.size)
        assertEquals(deadlines, reviews.findByRequestIdOrderByIdAsc(case.id).associate { it.id to it.deadline })
        assertConflict { service.verify(admin.id, case.id, command.copy(verificationNote = "changed")) }
        assertConflict { service.verify(user(Role.ADMIN).id, case.id, command) }
    }

    @Test fun `failedTransactionCannotHideWithoutHistory`() {
        val selected = photo(user())
        val case = create()
        val command = command(case, listOf(selected))
        tx.execute { doThrow(IllegalStateException("synthetic audit failure")).`when`(recorder).record(org.mockito.ArgumentMatchers.argThat<RemovalHistoryEvent> { it?.action == "VERIFIED" } ?: historyEvent(case.id)) }
        assertFails { service.verify(admin.id, case.id, command) }
        assertEquals(RemovalState.NONE, photos.findById(selected.id).orElseThrow().removalState)
        assertEquals(RemovalRequestStatus.DRAFT, requests.findById(case.id).orElseThrow().status)
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).isEmpty())
        assertTrue(reviews.findByRequestIdOrderByIdAsc(case.id).isEmpty())
        assertNull(commands.findByCommandId(command.commandId))
    }

    @Test fun `failed notice keeps pending review and retry sets only its real delivery deadline`() {
        val failedOwner = user(); val goodOwner = user()
        val case = create()
        val command = command(case, listOf(photo(failedOwner), photo(goodOwner)))
        tx.execute { doThrow(IllegalStateException("synthetic notice audit failure")).`when`(recorder).record(org.mockito.ArgumentMatchers.argThat<RemovalHistoryEvent> {
            it?.action == "NOTICE_STORED" && reviews.findById(it.reviewId!!).orElseThrow().ownerIdSnapshot == failedOwner.id
        } ?: historyEvent(case.id)) }
        val original = service.verify(admin.id, case.id, command)
        val pending = reviews.findByRequestIdOrderByIdAsc(case.id).single { it.ownerIdSnapshot == failedOwner.id }
        assertNull(pending.notifiedAt); assertNull(pending.deadline); assertNull(pending.notifiedMessageId)
        assertTrue(messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(failedOwner.id).isEmpty())
        assertEquals(now.plusSeconds(72 * 3600), reviews.findByRequestIdOrderByIdAsc(case.id).single { it.ownerIdSnapshot == goodOwner.id }.deadline)
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).all { it.state == RemovalItemState.VERIFIED })
        tx.execute { doCallRealMethod().`when`(recorder).record(org.mockito.ArgumentMatchers.any() ?: historyEvent(case.id)) }
        clock.set(now.plusSeconds(600))
        notifier.retryPending()
        val retried = reviews.findById(pending.id).orElseThrow()
        assertEquals(now.plusSeconds(600), retried.notifiedAt)
        assertEquals(now.plusSeconds(72 * 3600 + 600), retried.deadline)
        assertEquals(original, service.verify(admin.id, case.id, command))
    }

    @Test fun `suspended and missing owners route to admin without deadlines`() {
        val suspended = user().apply { suspendedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC) }
        users.saveAndFlush(suspended)
        val case = create()
        service.verify(admin.id, case.id, command(case, listOf(photo(suspended))))
        val unavailable = reviews.findByRequestIdOrderByIdAsc(case.id).single()
        assertNotNull(unavailable.unavailableReason)
        assertNull(unavailable.notifiedAt); assertNull(unavailable.deadline)
        assertTrue(messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(suspended.id).isEmpty())
        // A deleted live account still leaves its immutable review owner snapshot.
        val missingCase = create()
        val absentId = UUID.randomUUID()
        tx.execute {
            val request = requests.findForUpdate(missingCase.id)!!
            request.status = RemovalRequestStatus.OPEN
            reviews.saveAndFlush(PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = absentId))
        }
        notifier.notifyReview(reviews.findByRequestIdOrderByIdAsc(missingCase.id).single().id)
        assertNotNull(reviews.findByRequestIdOrderByIdAsc(missingCase.id).single().unavailableReason)
    }

    @Test fun `unknown foreign duplicate or unconfirmed selections cannot restrict`() {
        val selected = photo(user())
        val foreignEvent = events.saveAndFlush(Event(slug = "foreign-${UUID.randomUUID()}", name = "Foreign", date = LocalDate.of(2026, 10, 5), location = "Cebu", status = EventStatus.ACTIVE))
        val foreign = photos.saveAndFlush(Photo(eventId = foreignEvent.id, s3Key = "synthetic/foreign", pricePhp = BigDecimal("50")))
        val case = create()
        val valid = command(case, listOf(selected))
        for (bad in listOf(valid.copy(selectedPhotos = listOf(VerifyRemovalSelection(UUID.randomUUID(), RemovalMatchSource.MANUAL, "note"))),
            valid.copy(selectedPhotos = listOf(VerifyRemovalSelection(foreign.id, RemovalMatchSource.MANUAL, "note"))),
            valid.copy(selectedPhotos = valid.selectedPhotos + valid.selectedPhotos), valid.copy(selectedPhotos = emptyList()),
            valid.copy(selectedPhotos = listOf(VerifyRemovalSelection(selected.id, RemovalMatchSource.MANUAL))),
            valid.copy(verificationMethod = ""), valid.copy(verificationNote = ""), valid.copy(expectedVersion = 99))) {
            assertFailsWith<ApiException> { service.verify(admin.id, case.id, bad.copy(commandId = UUID.randomUUID())) }
        }
        assertEquals(RemovalState.NONE, photos.findById(selected.id).orElseThrow().removalState)
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).isEmpty())
    }

    @Test fun `selfie selection requires current version and evidence generation candidate and snapshots score`() {
        val selected = photo(user())
        val case = create()
        val selection = VerifyRemovalSelection(selected.id, RemovalMatchSource.SELFIE)
        val base = command(case, listOf(selected)).copy(selectedPhotos = listOf(selection))
        assertFailsWith<ApiException> { service.verify(admin.id, case.id, base) }
        tx.execute {
            val request = requests.findForUpdate(case.id)!!
            request.evidenceKey = "synthetic/case-selfie"
            request.evidenceGeneration = 2
            request.candidateCache = RemovalCandidateCache(request.version + 1, 1, listOf(RemovalCachedCandidate(selected.id, RemovalMatchSource.SELFIE, 0.91)))
            requests.flush()
        }
        val version = queries.adminDetail(admin.id, case.id).version
        assertFailsWith<ApiException> { service.verify(admin.id, case.id, base.copy(commandId = UUID.randomUUID(), expectedVersion = version)) }
        tx.execute {
            val request = requests.findForUpdate(case.id)!!
            request.candidateCache = RemovalCandidateCache(request.version + 1, 2, listOf(RemovalCachedCandidate(selected.id, RemovalMatchSource.SELFIE, 0.91)))
            requests.flush()
        }
        service.verify(admin.id, case.id, base.copy(commandId = UUID.randomUUID(), expectedVersion = queries.adminDetail(admin.id, case.id).version))
        assertEquals(0.91, items.findByRequestIdOrderByIdAsc(case.id).single().matchScore)
    }

    @Test fun `new command cannot expand or reset frozen verification`() {
        val owner = user(); val selected = photo(owner); val extra = photo(owner)
        val case = create()
        service.verify(admin.id, case.id, command(case, listOf(selected)))
        val current = queries.adminDetail(admin.id, case.id)
        val deadline = current.reviews.single().deadline
        assertConflict { service.verify(admin.id, case.id, command(current, listOf(selected, extra))) }
        assertEquals(listOf(selected.id), items.findByRequestIdOrderByIdAsc(case.id).map { it.photoIdSnapshot })
        assertEquals(deadline, queries.adminDetail(admin.id, case.id).reviews.single().deadline)
        assertEquals(RemovalState.NONE, photos.findById(extra.id).orElseThrow().removalState)
    }

    @Test fun `inbox removal retains audit notice deadline and deduplication`() {
        val owner = user(); val case = create()
        service.verify(admin.id, case.id, command(case, listOf(photo(owner))))
        val review = reviews.findByRequestIdOrderByIdAsc(case.id).single()
        inbox.markRead(owner.id, review.notifiedMessageId!!)
        inbox.remove(owner.id, review.notifiedMessageId!!)
        notifier.notifyReview(review.id)
        assertEquals(1, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(owner.id).size)
        assertEquals(review.deadline, reviews.findById(review.id).orElseThrow().deadline)
        assertEquals(1, history.findByRequestIdOrderByOccurredAtAscIdAsc(case.id, PageRequest.of(0, 100)).count { it.action == "NOTICE_STORED" })
    }

    @Test fun `related case intake note is preserved in private creation history`() {
        val owner = user(); val original = create()
        service.verify(admin.id, original.id, command(original, listOf(photo(owner))))
        val before = queries.adminDetail(admin.id, original.id)
        val relatedCommand = CreateRemovalRequestCommand(event.id, before.supportReference, "private contact", reason = "private reason",
            commandId = UUID.randomUUID(), adminNote = "Later discovery related to case ${original.id}")
        val related = service.create(admin.id, relatedCommand)
        assertEquals(before.supportReference, related.supportReference)
        assertEquals(relatedCommand.adminNote, history.findByRequestIdAndCommandId(related.id, relatedCommand.commandId).single().reason)
        assertEquals(before, queries.adminDetail(admin.id, original.id))
    }

    @Test fun `failed first page backs off so later pending owner receives notice`() {
        val selected = (1..51).map { photo(user()) }
        val case = create()
        tx.execute { doThrow(IllegalStateException("synthetic notice failure")).`when`(recorder).record(org.mockito.ArgumentMatchers.argThat<RemovalHistoryEvent> { it?.action == "NOTICE_STORED" } ?: historyEvent(case.id)) }
        service.verify(admin.id, case.id, command(case, selected))
        val ordered = reviews.findByRequestIdOrderByIdAsc(case.id)
        assertEquals(51, ordered.size)
        val lastOwnerReview = ordered.last()
        clock.set(now.plusSeconds(600))
        notifier.retryPending()
        assertTrue(reviews.findByRequestIdOrderByIdAsc(case.id).all { it.notifiedAt == null })
        tx.execute { doCallRealMethod().`when`(recorder).record(org.mockito.ArgumentMatchers.any() ?: historyEvent(case.id)) }
        clock.set(now.plusSeconds(660))
        notifier.retryPending()
        assertEquals(now.plusSeconds(660), reviews.findById(lastOwnerReview.id).orElseThrow().notifiedAt)
        assertTrue(ordered.take(50).all { reviews.findById(it.id).orElseThrow().notifiedAt == null }, "failed first page must respect retry backoff")
    }

    @Test fun `verification rejects enclosing transaction before persisting selection`() {
        val case = create(); val selected = photo(user())
        tx.execute {
            assertFailsWith<org.springframework.transaction.IllegalTransactionStateException> { service.verify(admin.id, case.id, command(case, listOf(selected))) }
            assertTrue(items.findByRequestIdOrderByIdAsc(case.id).isEmpty(), "split verification must reject an enclosing transaction before writes")
        }
    }

    @Test fun `waiting notifier cannot overwrite a committed notice after owner suspension`() {
        val owner = user(); val case = create()
        tx.execute { doThrow(IllegalStateException("synthetic initial notice failure")).`when`(recorder).record(org.mockito.ArgumentMatchers.argThat<RemovalHistoryEvent> { it?.action == "NOTICE_STORED" } ?: historyEvent(case.id)) }
        service.verify(admin.id, case.id, command(case, listOf(photo(owner))))
        val review = reviews.findByRequestIdOrderByIdAsc(case.id).single()
        tx.execute { doCallRealMethod().`when`(recorder).record(org.mockito.ArgumentMatchers.any() ?: historyEvent(case.id)) }
        val entered = java.util.concurrent.CountDownLatch(1)
        val release = java.util.concurrent.CountDownLatch(1)
        doAnswer {
            val result = java.util.Optional.ofNullable(entityManager.find(PhotoRemovalReview::class.java, review.id))
            if (Thread.currentThread().name == "stale-notifier") {
                entered.countDown()
                assertTrue(release.await(20, java.util.concurrent.TimeUnit.SECONDS))
            }
            result
        }.`when`(reviews).findById(review.id)
        val pool = java.util.concurrent.Executors.newSingleThreadExecutor { Thread(it, "stale-notifier") }
        try {
            val waiting = pool.submit { notifier.notifyReview(review.id) }
            val reachedRead = entered.await(20, java.util.concurrent.TimeUnit.SECONDS)
            if (!reachedRead && waiting.isDone) waiting.get(1, java.util.concurrent.TimeUnit.SECONDS)
            assertTrue(reachedRead)
            notifier.notifyReview(review.id)
            val delivered = reviews.findById(review.id).orElseThrow()
            assertEquals(now, delivered.notifiedAt)
            owner.suspendedAt = now.atOffset(ZoneOffset.UTC)
            users.saveAndFlush(owner)
            release.countDown()
            waiting.get(20, java.util.concurrent.TimeUnit.SECONDS)
            val after = reviews.findById(review.id).orElseThrow()
            assertEquals(delivered.notifiedMessageId, after.notifiedMessageId)
            assertEquals(delivered.notifiedAt, after.notifiedAt)
            assertEquals(delivered.deadline, after.deadline)
            assertNull(after.unavailableReason)
        } finally { release.countDown(); pool.shutdownNow() }
    }

    private var receiptCommandId: UUID? = null
    private fun command(case: PhotoRemovalRequestDto, photos: List<Photo>) = VerifyRemovalCommand(UUID.randomUUID().also { receiptCommandId = it },
        case.version, "Support identity verified", "private verification", photos.map { VerifyRemovalSelection(it.id, RemovalMatchSource.MANUAL, "Admin visually confirmed") })
    private fun create() = service.create(admin.id, CreateRemovalRequestCommand(event.id, "support-${UUID.randomUUID()}", "private contact", reason = "private reason", commandId = UUID.randomUUID()))
    private fun user(role: Role = Role.PHOTOGRAPHER) = users.saveAndFlush(User(email = "verify-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic $role", role = role))
    private fun photo(owner: User?) = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner?.id, s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("50")))
    private fun historyEvent(id: UUID) = RemovalHistoryEvent(requestId = id, actorType = RemovalActorType.SYSTEM, action = "NOTICE_STORED", occurredAt = now, commandId = UUID.randomUUID())
    private fun assertConflict(action: () -> Unit) { assertEquals(409, assertFailsWith<ApiException>(block = action).status.value()) }
    class MutableClock(initial: Instant) : Clock() {
        private val current = AtomicReference(initial)
        fun set(value: Instant) { current.set(value) }
        override fun getZone(): ZoneId = ZoneOffset.UTC
        override fun withZone(zone: ZoneId): Clock = Clock.fixed(current.get(), zone)
        override fun instant(): Instant = current.get()
    }

    @TestConfiguration
    class ClockConfig {
        @Bean
        @Primary
        fun removalVerificationClock(): MutableClock = MutableClock(Instant.parse("2026-10-05T00:00:00Z"))
    }

    companion object { private val UUID_REGEX = Regex("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}") }
}
