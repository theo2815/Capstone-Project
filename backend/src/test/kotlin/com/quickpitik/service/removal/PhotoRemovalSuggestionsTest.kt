package com.quickpitik.service.removal

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.removal.ReviewSuggestionsCommand
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito.*
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionDefinition
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.*

// RD-6: an admin promotes, dismisses or demotes the items of an open runner case.
class PhotoRemovalSuggestionsTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val commands = mock(PhotoRemovalCommandRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val photos = mock(PhotoRepository::class.java)
    private val reviews = mock(PhotoRemovalReviewRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val notifier = mock(PhotoRemovalNotifier::class.java)
    private val availability = mock(PhotoAvailabilityService::class.java)
    private val queries = mock(PhotoRemovalQueryService::class.java)
    private val recorder = mock(PhotoRemovalHistoryRecorder::class.java)
    private val audit = mock(PhotoRemovalHistoryRepository::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val mapper = jacksonObjectMapper().findAndRegisterModules()
    private val now = Instant.parse("2026-11-12T03:00:00Z")
    private val adminId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private val ownerA = UUID.randomUUID()
    private val ownerB = UUID.randomUUID()
    private val request = PhotoRemovalRequest(eventId = eventId, eventIdSnapshot = eventId, eventNameSnapshot = "Cebu run",
        requesterUserId = UUID.randomUUID(), requesterContact = "runner@test.local", supportReference = "Runner request",
        requesterReason = "r", publicToOwnerReason = "Privacy removal request", creatorId = UUID.randomUUID(),
        creationCommandId = UUID.randomUUID(), source = RemovalRequestSource.RUNNER, status = RemovalRequestStatus.OPEN,
        livenessBand = RunnerLivenessBand.PASS)
    private val savedReviews = mutableListOf<PhotoRemovalReview>()
    private val allItems = mutableListOf<PhotoRemovalItem>()
    private val allPhotos = mutableMapOf<UUID, Photo>()
    private var receipt: PhotoRemovalCommand? = null

    private val service = PhotoRemovalService(requests, commands, mock(EventRepository::class.java), users, items, photos, reviews,
        notifier, availability, mock(PhotoRemovalEvidenceService::class.java), queries, recorder, PhotoRemovalTransitions(),
        TransactionTemplate(manager), mapper, Clock.fixed(now, ZoneOffset.UTC), mock(org.springframework.context.ApplicationEventPublisher::class.java),
        mock(com.quickpitik.service.photos.PhotoErasureService::class.java), mock(PhotoErasureJobRepository::class.java),
        mock(com.quickpitik.service.orders.OrderPhotoRemovalService::class.java), mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java),
        audit, com.quickpitik.config.PhotoRemovalProperties(), mock(RunnerAutoHidePolicy::class.java))

    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenReturn(SimpleTransactionStatus())
        `when`(users.findById(adminId)).thenReturn(Optional.of(User(email = "admin@test.local", passwordHash = "x", name = "Admin", role = Role.ADMIN)))
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenAnswer { allItems.toList() }
        `when`(reviews.findByRequestIdOrderByIdAsc(request.id)).thenAnswer { savedReviews.toList() }
        `when`(reviews.saveAndFlush(anyArg<PhotoRemovalReview>())).thenAnswer { inv -> inv.getArgument<PhotoRemovalReview>(0).also { savedReviews += it } }
        `when`(photos.findAllByIdForUpdate(anyArg())).thenAnswer { inv -> inv.getArgument<Collection<UUID>>(0).mapNotNull { allPhotos[it] } }
        `when`(queries.adminDetail(adminId, request.id)).thenAnswer { request.toAdminDto(allItems, savedReviews) }
        `when`(commands.findByCommandId(anyArg())).thenAnswer { receipt }
        `when`(commands.insertClaim(anyArg(), anyArg(), anyString(), anyArg(), anyString(), isNull(), anyArg())).thenAnswer { inv ->
            if (receipt != null) 0 else {
                receipt = PhotoRemovalCommand(inv.getArgument(0), request.id, RemovalCommandOperation.valueOf(inv.getArgument(2)), adminId,
                    inv.getArgument(4), claimedAt = now)
                1
            }
        }
    }

    private fun <T> anyArg(): T = org.mockito.ArgumentMatchers.any<T>() ?: null as T

    private fun photo(owner: UUID, state: RemovalState = RemovalState.NONE, event: UUID = eventId) = Photo(eventId = event,
        photographerId = owner, s3Key = "p/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal.TEN).apply { removalState = state }
            .also { allPhotos[it.id] = it }

    private fun item(photo: Photo?, state: RemovalItemState, reviewId: UUID? = null, actor: UUID? = null, decision: RemovalDecision? = null) =
        PhotoRemovalItem(requestId = request.id, photoIdSnapshot = photo?.id ?: UUID.randomUUID(), photoId = photo?.id,
            ownerIdSnapshot = photo?.photographerId, state = state, matchSource = RemovalMatchSource.SELFIE, matchScore = 0.9,
            reviewId = reviewId, verificationActorId = actor, verifiedAt = if (state == RemovalItemState.VERIFIED) now else null,
            adminDecision = decision).also { allItems += it }

    private fun command(promote: List<PhotoRemovalItem> = emptyList(), dismiss: List<PhotoRemovalItem> = emptyList(),
                        demote: List<PhotoRemovalItem> = emptyList(), id: UUID = UUID.randomUUID()) =
        ReviewSuggestionsCommand(id, request.version, promote.map { it.id }, dismiss.map { it.id }, demote.map { it.id })

    private fun history(): List<RemovalHistoryEvent> {
        val captor = ArgumentCaptor.forClass(RemovalHistoryEvent::class.java)
        verify(recorder, atLeast(0)).record(captor.capture() ?: RemovalHistoryEvent(request.id, actorType = RemovalActorType.ADMIN,
            action = "x", occurredAt = now, commandId = UUID.randomUUID()))
        return captor.allValues
    }

    private fun refused(c: ReviewSuggestionsCommand) = assertFailsWith<ApiException> { service.reviewSuggestions(adminId, request.id, c) }

    @Test fun `promote hides the photo and creates one review per new owner`() {
        val a1 = item(photo(ownerA), RemovalItemState.CANDIDATE)
        val a2 = item(photo(ownerA), RemovalItemState.CANDIDATE)
        val b = item(photo(ownerB), RemovalItemState.CANDIDATE)
        service.reviewSuggestions(adminId, request.id, command(promote = listOf(a1, a2, b)))
        assertEquals(setOf(ownerA, ownerB), savedReviews.map { it.ownerIdSnapshot }.toSet())
        assertEquals(2, savedReviews.size)
        for (i in listOf(a1, a2, b)) {
            assertEquals(RemovalItemState.VERIFIED to adminId, i.state to i.verificationActorId)
            assertEquals(savedReviews.single { it.ownerIdSnapshot == i.ownerIdSnapshot }.id, i.reviewId)
        }
        verify(availability).refreshRemovalState(setOf(a1.photoId!!, a2.photoId!!, b.photoId!!))
        assertEquals(3, history().count { it.action == RemovalHistoryActions.VERIFIED && it.actorType == RemovalActorType.ADMIN })
        savedReviews.forEach { verify(notifier).attempt(it.id) }
    }

    @Test fun `promote into an already-notified review re-notifies and restarts the 72-hour clock`() {
        val notified = PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = ownerA, ownerId = ownerA, notifiedMessageId = UUID.randomUUID(),
            notifiedAt = now.minusSeconds(80 * 3600), deadline = now.minusSeconds(8 * 3600), overdue = true, overdueAt = now.minusSeconds(3600))
            .also { savedReviews += it }
        item(photo(ownerA), RemovalItemState.VERIFIED, notified.id) // the auto-hidden photo the owner was told about
        val c = item(photo(ownerA), RemovalItemState.CANDIDATE)
        request.version = 4
        service.reviewSuggestions(adminId, request.id, command(promote = listOf(c)))
        assertEquals(notified.id, c.reviewId)
        assertNull(notified.notifiedAt); assertNull(notified.notifiedMessageId); assertNull(notified.deadline)
        assertFalse(notified.overdue); assertNull(notified.overdueAt); assertEquals(now, notified.notificationNextAttemptAt)
        val row = history().single { it.itemId == c.id }
        assertEquals("owner_reverification", row.metadata["kind"])
        assertEquals("5", row.metadata["noticeVersion"])
        verify(notifier).attempt(notified.id)
        assertEquals(1, savedReviews.size) // no second review for the same owner
    }

    @Test fun `re-promoting a photo demoted after the owner's notice keeps the review and sends no new notice`() {
        val notifiedAt = now.minusSeconds(3600)
        val deadline = now.plusSeconds(71 * 3600)
        val notified = PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = ownerA, ownerId = ownerA, notifiedMessageId = UUID.randomUUID(),
            notifiedAt = notifiedAt, deadline = deadline).also { savedReviews += it }
        val c = item(photo(ownerA), RemovalItemState.CANDIDATE) // auto-hidden, listed in that notice, then demoted
        `when`(audit.existsByRequestIdAndItemIdAndActionAndOccurredAtAfter(request.id, c.id,RemovalHistoryActions.SUGGESTED, notifiedAt)).thenReturn(true)
        service.reviewSuggestions(adminId, request.id, command(promote = listOf(c)))
        assertEquals(notified.id, c.reviewId)
        assertEquals(notifiedAt to deadline, notified.notifiedAt to notified.deadline)
        assertNull(history().single { it.itemId == c.id }.metadata["kind"])
        verify(notifier, never()).attempt(notified.id)
    }

    @Test fun `dismiss leaves the photo public`() {
        val c = item(photo(ownerA), RemovalItemState.CANDIDATE)
        service.reviewSuggestions(adminId, request.id, command(dismiss = listOf(c)))
        assertEquals(RemovalItemState.EXCLUDED, c.state)
        assertEquals(RemovalState.NONE, allPhotos.getValue(c.photoId!!).removalState)
        verify(availability, never()).refreshRemovalState(anyArg())
        assertEquals(RemovalHistoryActions.SUGGESTION_DISMISSED, history().single().action)
        verifyNoInteractions(notifier)
    }

    @Test fun `demote makes an auto-hidden photo public and keeps the case open`() {
        val review = PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = ownerA, ownerId = ownerA).also { savedReviews += it }
        val hidden = item(photo(ownerA, RemovalState.REVIEW), RemovalItemState.VERIFIED, review.id)
        service.reviewSuggestions(adminId, request.id, command(demote = listOf(hidden)))
        assertEquals(RemovalItemState.CANDIDATE to null, hidden.state to hidden.reviewId)
        verify(availability).refreshRemovalState(setOf(hidden.photoId!!))
        assertEquals(RemovalRequestStatus.OPEN, request.status)
        val row = history().single()
        assertEquals(RemovalHistoryActions.SUGGESTED to "demoted", row.action to row.metadata["kind"])
    }

    @Test fun `an admin-decided item cannot be demoted`() {
        val decided = item(photo(ownerA), RemovalItemState.VERIFIED, decision = RemovalDecision.MORE_EVIDENCE)
        val adminVerified = item(photo(ownerA), RemovalItemState.VERIFIED, actor = adminId)
        for (i in listOf(decided, adminVerified)) assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, refused(command(demote = listOf(i))).code)
        assertNull(receipt)
    }

    @Test fun `a VERIFIED item cannot be dismissed`() {
        val v = item(photo(ownerA), RemovalItemState.VERIFIED)
        val ex = refused(command(dismiss = listOf(v)))
        assertEquals(409 to ErrorCodes.INVALID_STATE_TRANSITION, ex.status.value() to ex.code)
        assertEquals(RemovalItemState.VERIFIED, v.state)
        // An id from another case is refused the same way.
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, refused(ReviewSuggestionsCommand(UUID.randomUUID(), 0, dismiss = listOf(UUID.randomUUID()))).code)
    }

    @Test fun `a replayed command id changes nothing`() {
        val c = item(photo(ownerA), RemovalItemState.CANDIDATE)
        val first = command(dismiss = listOf(c))
        val result = service.reviewSuggestions(adminId, request.id, first)
        assertEquals(result, service.reviewSuggestions(adminId, request.id, first))
        verify(commands, times(1)).insertClaim(anyArg(), anyArg(), anyString(), anyArg(), anyString(), isNull(), anyArg())
        assertEquals(1, history().size)
        assertEquals(RemovalCommandOperation.REVIEW_SUGGESTIONS, receipt!!.operation)
    }

    @Test fun `promoting a deleted or removed photo is refused`() {
        val gone = item(null, RemovalItemState.CANDIDATE)
        val removed = item(photo(ownerA, RemovalState.REMOVED), RemovalItemState.CANDIDATE)
        val moved = item(photo(ownerA, event = UUID.randomUUID()), RemovalItemState.CANDIDATE)
        for (i in listOf(gone, removed, moved)) {
            assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, refused(command(promote = listOf(i))).code)
            assertEquals(RemovalItemState.CANDIDATE, i.state)
        }
        assertNull(receipt)
    }

    @Test fun `an empty or closed review is refused`() {
        assertEquals(400, refused(command()).status.value())
        val c = item(photo(ownerA), RemovalItemState.CANDIDATE)
        request.status = RemovalRequestStatus.DRAFT
        assertEquals(409, refused(command(dismiss = listOf(c))).status.value())
    }
}
