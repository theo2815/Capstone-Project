package com.quickpitik.service.removal

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.config.RekognitionProperties
import com.quickpitik.dto.photos.FaceCandidate
import com.quickpitik.dto.photos.FaceCandidateResult
import com.quickpitik.dto.removal.SearchImage
import com.quickpitik.dto.removal.SubmitRunnerRemovalCommand
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.LivenessResult
import com.quickpitik.service.ai.RekognitionIdentityClient
import com.quickpitik.service.liveness.LivenessSessionService
import com.quickpitik.service.photos.PhotoSearchService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito.*
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RunnerPhotoRemovalSubmitTest {
    private val eligibility = mock(RunnerRemovalEligibilityService::class.java)
    private val liveness = mock(LivenessSessionService::class.java)
    private val sessions = mock(LivenessSessionRepository::class.java)
    private val identity = mock(RekognitionIdentityClient::class.java)
    @Suppress("UNCHECKED_CAST")
    private val identityProvider = mock(ObjectProvider::class.java) as ObjectProvider<RekognitionIdentityClient>
    private val removals = mock(PhotoRemovalService::class.java)
    private val evidence = mock(PhotoRemovalEvidenceService::class.java)
    private val search = mock(PhotoSearchService::class.java)
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val photos = mock(PhotoRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val events = mock(EventRepository::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val props = PhotoRemovalProperties()
    private val policy = RunnerAutoHidePolicy(props, RekognitionProperties())

    private val runner = User(email = "runner@test.local", passwordHash = "x", name = "Runner", role = Role.RUNNER,
        emailVerifiedAt = OffsetDateTime.now())
    private val event = Event(slug = "run", name = "Cebu run", date = LocalDate.of(2026, 11, 1), location = "Cebu", status = EventStatus.ACTIVE)
    private val key = UUID.randomUUID()
    private val command = SubmitRunnerRemovalCommand("sess-1", event.id, "PRIVACY", "please")
    private val jpeg = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 1, 2, 3)
    private val draft = RunnerDraft(UUID.randomUUID(), "photo-removal/x/evidence/$key")

    private val service = RunnerPhotoRemovalService(
        eligibility, liveness, LegalProperties("1.3"), props,
        Clock.fixed(Instant.parse("2026-11-10T04:00:00Z"), ZoneOffset.UTC),
        sessions, identityProvider, policy, removals, evidence, search, requests, items, photos, users, events,
        TransactionTemplate(manager),
    )

    init {
        `when`(manager.getTransaction(any())).thenReturn(SimpleTransactionStatus())
        `when`(identityProvider.getIfAvailable()).thenReturn(identity)
        `when`(eligibility.check(runner.id, event.id)).thenReturn(RunnerRemovalEligibility(true, null, 2, true, null))
        `when`(liveness.claim(runner.id, "sess-1", LivenessPurpose.RUNNER_PHOTO_REMOVAL, event.id)).thenReturn(true)
        `when`(sessions.findBySessionIdAndUserIdAndPurpose("sess-1", runner.id, LivenessPurpose.RUNNER_PHOTO_REMOVAL)).thenReturn(
            LivenessSession("sess-1", runner.id, LivenessPurpose.RUNNER_PHOTO_REMOVAL, event.id, "1.3", OffsetDateTime.parse("2026-11-10T03:59:00Z")))
        `when`(users.findById(runner.id)).thenReturn(Optional.of(runner))
        `when`(events.findById(event.id)).thenReturn(Optional.of(event))
        `when`(removals.openRunnerDraft(anyArg() ?: draftCommand())).thenReturn(draft)
        `when`(removals.completeRunnerCase(anyArg() ?: draft.requestId, anyArg() ?: RunnerMatchOutcome(false, emptyList(), false, true, false))).thenReturn(RunnerCaseResult(emptyList(), false, 0, 1))
        `when`(requests.findById(draft.requestId)).thenReturn(Optional.of(runnerCase(draft.requestId)))
    }

    private fun runnerCase(id: UUID = UUID.randomUUID(), status: RemovalRequestStatus = RemovalRequestStatus.OPEN) =
        PhotoRemovalRequest(id = id, eventId = event.id, eventIdSnapshot = event.id, eventNameSnapshot = event.name,
            requesterUserId = runner.id, requesterContact = runner.email, supportReference = "Runner request", requesterReason = "r",
            publicToOwnerReason = "p", creatorId = runner.id, creationCommandId = key, source = RemovalRequestSource.RUNNER,
            status = status, livenessBand = RunnerLivenessBand.PASS)

    private fun <T> anyArg(): T = any()
    private fun passed(confidence: Float = 97f, image: ByteArray? = jpeg) =
        `when`(identity.livenessResult("sess-1")).thenReturn(LivenessResult(LivenessSessionStatus.SUCCEEDED, confidence, image))
    private fun photo(removal: RemovalState = RemovalState.NONE) = Photo(eventId = event.id, photographerId = UUID.randomUUID(),
        s3Key = "p/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal.TEN).apply {
        removalState = removal
        facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = "${event.id}.$id"))
    }
    private fun matches(vararg personIds: String) = `when`(search.findFaceCandidates(anyArg() ?: event.id, anyArg() ?: SearchImage(jpeg, "image/jpeg", "x"), anyInt()))
        .thenReturn(FaceCandidateResult(personIds.map { FaceCandidate(it, 0.995) }, 50, false))
    private fun outcome(): RunnerMatchOutcome {
        val captor = ArgumentCaptor.forClass(RunnerMatchOutcome::class.java)
        verify(removals).completeRunnerCase(anyArg() ?: draft.requestId, captor.capture() ?: RunnerMatchOutcome(false, emptyList(), false, true, false))
        return captor.value
    }

    @Test
    fun `same idempotency key returns the first result without a second AWS read`() {
        val first = runnerCase()
        `when`(requests.findByCreationCommandId(key)).thenReturn(first)
        `when`(items.findByRequestIdOrderByIdAsc(first.id)).thenReturn(listOf(
            PhotoRemovalItem(requestId = first.id, photoIdSnapshot = UUID.randomUUID(), state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.SELFIE),
            PhotoRemovalItem(requestId = first.id, photoIdSnapshot = UUID.randomUUID(), state = RemovalItemState.CANDIDATE, matchSource = RemovalMatchSource.SELFIE)))
        val result = service.submit(runner.id, key, command)
        assertEquals(first.id, result.requestId)
        assertEquals("PENDING_ADMIN_REVIEW", result.status)
        assertEquals(1 to 1, result.hiddenCount to result.suggestedCount)
        verifyNoInteractions(identity, liveness, removals, evidence)

        // Someone else's key is a conflict, never their case.
        val ex = assertFailsWith<ApiException> { service.submit(UUID.randomUUID(), key, command) }
        assertEquals(ErrorCodes.CONFLICT, ex.code)
    }

    @Test
    fun `a photographer-purpose or other-event session is refused before any AWS read`() {
        `when`(liveness.claim(runner.id, "sess-1", LivenessPurpose.RUNNER_PHOTO_REMOVAL, event.id)).thenReturn(false)
        val ex = assertFailsWith<ApiException> { service.submit(runner.id, key, command) }
        assertEquals(ErrorCodes.LIVENESS_SESSION_INVALID, ex.code)
        verifyNoInteractions(identity, removals, evidence)
    }

    @Test
    fun `a failed liveness band files nothing and stores nothing`() {
        passed(confidence = 20f)
        val ex = assertFailsWith<ApiException> { service.submit(runner.id, key, command) }
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY to ErrorCodes.LIVENESS_FAILED, ex.status to ex.code)
        `when`(identity.livenessResult("sess-1")).thenReturn(LivenessResult(LivenessSessionStatus.FAILED, 99f, jpeg))
        assertEquals(ErrorCodes.LIVENESS_FAILED, assertFailsWith<ApiException> { service.submit(runner.id, key, command) }.code)
        verifyNoInteractions(removals, evidence, search)
    }

    @Test
    fun `an uncertain band is recorded on the draft`() {
        passed(confidence = 70f)
        matches()
        service.submit(runner.id, key, command)
        val captor = ArgumentCaptor.forClass(RunnerDraftCommand::class.java)
        verify(removals).openRunnerDraft(captor.capture() ?: draftCommand())
        assertEquals(RunnerLivenessBand.UNCERTAIN, captor.value.livenessBand)
        assertEquals("1.3", captor.value.consentVersion)
        assertEquals(Instant.parse("2026-11-10T03:59:00Z"), captor.value.consentAt)
    }

    @Test
    fun `a foreign event's person id is ignored`() {
        passed()
        val own = photo()
        val foreign = "${UUID.randomUUID()}.${UUID.randomUUID()}"
        matches("${event.id}.${own.id}", foreign)
        `when`(photos.findRemovalReviewCandidates(event.id, setOf("${event.id}.${own.id}"))).thenReturn(listOf(own))
        service.submit(runner.id, key, command)
        verify(photos).findRemovalReviewCandidates(event.id, setOf("${event.id}.${own.id}"))
        assertEquals(listOf(own.id), outcome().candidates.map { it.photoId })
    }

    @Test
    fun `an AI error after a pass files a DRAFT case with the evidence and MATCHING_UNAVAILABLE`() {
        passed()
        `when`(search.findFaceCandidates(anyArg() ?: event.id, anyArg() ?: SearchImage(jpeg, "image/jpeg", "x"), anyInt()))
            .thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "down"))
        service.submit(runner.id, key, command)
        verify(evidence).putSystemEvidence((eq(draft.evidenceKey) ?: draft.evidenceKey), anyArg() ?: SearchImage(jpeg, "image/jpeg", "x"))
        assertTrue(outcome().matchingUnavailable)
    }

    @Test
    fun `a unique-index conflict answers REMOVAL_REQUEST_OPEN before any upload`() {
        passed()
        `when`(removals.openRunnerDraft(anyArg() ?: draftCommand()))
            .thenThrow(ApiException(HttpStatus.CONFLICT, ErrorCodes.REMOVAL_REQUEST_OPEN, "open"))
        assertEquals(ErrorCodes.REMOVAL_REQUEST_OPEN, assertFailsWith<ApiException> { service.submit(runner.id, key, command) }.code)
        verifyNoInteractions(evidence, search)
    }

    @Test
    fun `the evidence key is recorded before the upload`() {
        passed()
        matches()
        service.submit(runner.id, key, command)
        val order = inOrder(removals, evidence)
        order.verify(removals).openRunnerDraft(anyArg() ?: draftCommand())
        order.verify(evidence).putSystemEvidence((eq(draft.evidenceKey) ?: draft.evidenceKey), anyArg() ?: SearchImage(jpeg, "image/jpeg", "x"))
        order.verify(removals).completeRunnerCase(anyArg() ?: draft.requestId, anyArg() ?: RunnerMatchOutcome(false, emptyList(), false, true, false))
    }

    @Test
    fun `a missing reference image stores nothing`() {
        passed(image = null)
        assertEquals(ErrorCodes.LIVENESS_UNAVAILABLE, assertFailsWith<ApiException> { service.submit(runner.id, key, command) }.code)
        verifyNoInteractions(removals, evidence)
    }

    @Test
    fun `an already-removed photo is never added`() {
        passed()
        val kept = photo()
        val gone = photo(RemovalState.REMOVED)
        matches("${event.id}.${kept.id}", "${event.id}.${gone.id}")
        `when`(photos.findRemovalReviewCandidates(anyArg() ?: event.id, anyArg() ?: emptySet())).thenReturn(listOf(kept, gone))
        service.submit(runner.id, key, command)
        assertEquals(listOf(kept.id), outcome().candidates.map { it.photoId })
    }

    @Test
    fun `the client never supplies photo ids or an image`() {
        assertEquals(setOf("sessionId", "eventId", "reasonCode", "details"),
            SubmitRunnerRemovalCommand::class.java.declaredFields.map { it.name }.toSet())
    }

    @Test
    fun `an unknown reason code is refused before the session is used`() {
        val ex = assertFailsWith<ApiException> { service.submit(runner.id, key, command.copy(reasonCode = "BORED")) }
        assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
        verifyNoInteractions(liveness, identity)
    }

    private fun draftCommand() = RunnerDraftCommand(UUID.randomUUID(), runner.id, runner.email, event, key, "PRIVACY", null, "1.3",
        Instant.EPOCH, "sess-1", RunnerLivenessBand.PASS, BigDecimal.ONE, "image/jpeg")
}
