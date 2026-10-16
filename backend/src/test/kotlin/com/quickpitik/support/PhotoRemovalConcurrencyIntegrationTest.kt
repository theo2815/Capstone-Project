package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.photos.PhotoAvailabilityService
import com.quickpitik.service.photos.PhotoWorkFence
import com.quickpitik.service.photos.PhotoWorkKind
import com.quickpitik.service.photographer.PhotographerPhotoService
import com.quickpitik.service.admin.AdminFlagService
import com.quickpitik.exception.ConflictException
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertAll
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.test.context.TestPropertySource
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.jdbc.core.JdbcTemplate
import java.math.BigDecimal
import java.time.Instant
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@TestPropertySource(properties = ["app.admin.flags-enabled=true", "app.ai-api.indexing-mode=batch",
    "app.ai-api.webhook.poll-grace-seconds=600"])
class PhotoRemovalConcurrencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var photographerSettings: PhotographerSettingsRepository
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var availability: PhotoAvailabilityService
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var manager: PhotographerPhotoService
    @Autowired lateinit var flags: FlagRepository
    @Autowired lateinit var moderation: AdminFlagService
    @org.springframework.test.context.bean.override.mockito.MockitoSpyBean lateinit var fence: PhotoWorkFence
    @org.springframework.test.context.bean.override.mockito.MockitoSpyBean lateinit var clock: java.time.Clock
    @org.springframework.test.context.bean.override.mockito.MockitoSpyBean lateinit var aiProperties: com.quickpitik.config.AiProperties
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var dataSource: javax.sql.DataSource
    @Autowired lateinit var context: org.springframework.context.ApplicationContext
    @Autowired lateinit var batchIndexer: com.quickpitik.service.photographer.PhotoBatchIndexingService
    @Autowired lateinit var batchIngest: com.quickpitik.service.photographer.PhotoBatchIngestService
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var provider: AiApiClient

    private lateinit var owner: User
    private lateinit var event: Event
    private lateinit var photo: Photo

    @BeforeEach fun fixture() {
        owner = users.saveAndFlush(User(email = "composition-${UUID.randomUUID()}@test.local",
            passwordHash = "unused", name = "Photographer", role = Role.PHOTOGRAPHER))
        // Photo restore needs an approved photographer (security audit 2026-10-14, D6).
        photographerSettings.saveAndFlush(PhotographerSettings(userId = owner.id, verificationStatus = VerificationStatus.APPROVED))
        event = events.saveAndFlush(Event(slug = "composition-${UUID.randomUUID()}", name = "Cebu run",
            date = LocalDate.of(2026, 10, 6), location = "Cebu", status = EventStatus.ACTIVE))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "synthetic/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("125")))
    }

    private fun verifiedCase(): PhotoRemovalItem {
            val request = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id,
                eventIdSnapshot = event.id, eventNameSnapshot = event.name,
                requesterContact = "Support contact", supportReference = "Support-${UUID.randomUUID()}",
                requesterReason = "Privacy", publicToOwnerReason = "Verified privacy request",
                creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(),
                status = RemovalRequestStatus.OPEN))
            return items.saveAndFlush(PhotoRemovalItem(requestId = request.id, photoIdSnapshot = photo.id,
                photoId = photo.id, ownerIdSnapshot = owner.id, state = RemovalItemState.VERIFIED,
                matchSource = RemovalMatchSource.MANUAL))
    }

    @Test fun `closingOneCaseCannotReleaseAnother`() {
        val caseA = verifiedCase()
        verifiedCase()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertEquals(RemovalState.REVIEW, photos.findById(photo.id).orElseThrow().removalState)

        tx.executeWithoutResult {
            items.findById(caseA.id).orElseThrow().apply { state = RemovalItemState.RETAINED }
            requests.findById(caseA.requestId).orElseThrow().apply {
                status = RemovalRequestStatus.RESOLVED
                resolvedAt = Instant.now()
            }
            items.flush()
            availability.refreshRemovalState(setOf(photo.id))
        }
        val afterClosingOnlyCaseA = photos.findById(photo.id).orElseThrow()
        assertEquals(RemovalState.REVIEW, afterClosingOnlyCaseA.removalState)
    }

    @Test fun `flagDismissAndRestoreCannotPublishReview`() {
        verifiedCase()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        val flag = flags.saveAndFlush(Flag(targetKindWire = "photo", targetId = photo.id, reason = "privacy"))
        moderation.hide(owner.id, flag.id, "Review")
        manager.trash(owner.id, event.id, listOf(photo.id))
        moderation.dismiss(owner.id, flag.id, "Flag dismissed; removal review remains")

        val restoreError = runCatching { manager.restore(owner.id, event.id, listOf(photo.id)) }.exceptionOrNull()
        val afterFlagDismissAndOwnerRestore = photos.findById(photo.id).orElseThrow()
        assertAll(
            { assertEquals(RemovalState.REVIEW, afterFlagDismissAndOwnerRestore.removalState) },
            { kotlin.test.assertIs<ConflictException>(restoreError) },
            { assertNotNull(afterFlagDismissAndOwnerRestore.deletedAt) },
        )
    }

    @Test fun `lateProducerCannotUndoRemoval`() {
        // A watermark worker read PROCESSING before a separate admin transaction authorized removal.
        photo.status = PhotoStatus.PROCESSING
        photos.saveAndFlush(photo)
        val stale = photos.findById(photo.id).orElseThrow()
        val item = verifiedCase().apply {
            state = RemovalItemState.REMOVAL_AUTHORIZED
            adminDecision = RemovalDecision.REMOVE
            adminDecisionActorId = owner.id
            adminDecisionReason = "Verified removal decision"
            adminDecidedAt = Instant.now()
        }
        items.saveAndFlush(item)
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, owner.id) }
        val stalePublishUpdatedRows = tx.execute {
            photos.publishWatermarked(photo.id, "synthetic/late.jpg", 1L, 2L, 3L)
        }
        stale.altText = "Late writer"
        photos.saveAndFlush(stale)
        val reloadedAfterStaleEntitySave = photos.findById(photo.id).orElseThrow()
        val claimForRemovedPhoto = fence.claim(photo.id, PhotoWorkKind.WATERMARK)
        assertAll(
            { assertNull(claimForRemovedPhoto) },
            { assertEquals(0, stalePublishUpdatedRows) },
            { assertEquals(RemovalState.REMOVED, reloadedAfterStaleEntitySave.removalState) },
        )
    }

    @Test fun `normal hard delete cannot bypass an active review`() {
        verifiedCase()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertFailsWith<ConflictException> { manager.permanentDelete(owner.id, event.id, listOf(photo.id)) }
    }

    @Test fun `claims persist targets before writes and restrictions invalidate their generation`() {
        val claim = assertNotNull(fence.claim(photo.id, PhotoWorkKind.WATERMARK))
        assertNull(fence.claim(photo.id, PhotoWorkKind.WATERMARK))
        assertEquals("events/${event.id}/photos/${photo.id}/watermark.jpg",
            jdbc.queryForObject("SELECT storage_key FROM photo_work_outputs WHERE token = ?", String::class.java, claim.token))
        assertTrue(fence.isCurrent(claim))
        val item = verifiedCase()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertFalse(fence.isCurrent(claim))
        item.state = RemovalItemState.RETAINED
        items.saveAndFlush(item)
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertFalse(fence.isCurrent(claim), "Closing review must not revive an old writer")
        fence.finish(claim)
        assertNotNull(fence.claim(photo.id, PhotoWorkKind.WATERMARK))
    }

    @Test fun `expiry never proves an unresolved write ended or allows its deletion`() {
        val claim = assertNotNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        jdbc.update("UPDATE photos SET indexing_work_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE id = ?", photo.id)
        assertFalse(fence.isCurrent(claim))
        assertNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertEquals(listOf(photo.id), manager.permanentDelete(owner.id, event.id, listOf(photo.id)).failed)
        assertTrue(photos.existsById(photo.id))
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE token = ? AND finished_at IS NULL",
            Int::class.java, claim.token))
    }

    @Test fun `late person result remains recorded even when its claim is stale`() {
        val claim = assertNotNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        verifiedCase()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        fence.recordPerson(claim, "late-person")
        assertFalse(fence.isCurrent(claim))
        assertEquals("late-person", jdbc.queryForObject(
            "SELECT person_ids ->> 0 FROM photo_work_outputs WHERE token = ?", String::class.java, claim.token))
    }

    @Test fun `removed photos cannot be requeued for indexing or watermarking`() {
        val item = verifiedCase().apply {
            state = RemovalItemState.REMOVAL_AUTHORIZED
            adminDecision = RemovalDecision.REMOVE
            adminDecisionReason = "Verified removal"
            adminDecisionActorId = owner.id
        }
        items.saveAndFlush(item)
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, owner.id) }
        assertAll(
            { assertEquals(0, tx.execute { photos.resetForRewatermark(event.id) }) },
            { assertEquals(0, tx.execute { photos.requeueIndexing(event.id, listOf(IndexingStatus.PENDING)) }) },
        )
    }

    private fun <T> anyArg(): T = org.mockito.Mockito.any()

    @Test fun `batch writes have durable attribution outside transactions and late results cannot publish`() {
        val bibJob = UUID.randomUUID().toString()
        val faceJob = UUID.randomUUID().toString()
        var externalTransaction = false
        var recordedBeforeWrite = false
        org.mockito.Mockito.`when`(storage.getBytes(photo.s3Key)).thenReturn(byteArrayOf(1))
        org.mockito.Mockito.`when`(provider.bibsRecognizeMega(anyArg())).thenAnswer {
            externalTransaction = externalTransaction || org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive()
            recordedBeforeWrite = jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE photo_id = ? AND batch_id IS NOT NULL",
                Int::class.java, photo.id) == 1
            com.quickpitik.dto.ai.JobCreateResult(bibJob)
        }
        org.mockito.Mockito.`when`(provider.facesEnrollMega(anyArg(), anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(faceJob))
        assertTrue(batchIndexer.drainEvent(event.id, java.time.OffsetDateTime.now().plusMinutes(1)))
        verifiedCase()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        org.mockito.Mockito.`when`(provider.jobStatus(faceJob, 0, 1)).thenAnswer {
            externalTransaction = externalTransaction || org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive()
            com.quickpitik.dto.ai.JobStatusResult(job_id = faceJob, status = "completed",
                result = listOf(mapOf("ref" to photo.id.toString(), "person_id" to "late-batch-person")))
        }
        batchIngest.ingest(faceJob)
        val fresh = photos.findById(photo.id).orElseThrow()
        assertAll(
            { assertFalse(externalTransaction, "External batch I/O must not hold a transaction") },
            { assertTrue(recordedBeforeWrite, "Record claim and batch before submission") },
            { assertTrue(fresh.facePersons.isEmpty(), "Stale batch result must not attach to photo") },
            { assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE photo_id = ? AND person_ids @> '[\"late-batch-person\"]'::jsonb",
                Int::class.java, photo.id)) },
        )
    }

    @Test fun `unknown face batch submission retains claims and known bib job`() {
        val bibJob = UUID.randomUUID().toString()
        org.mockito.Mockito.`when`(storage.getBytes(photo.s3Key)).thenReturn(byteArrayOf(1))
        org.mockito.Mockito.`when`(provider.bibsRecognizeMega(anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(bibJob))
        org.mockito.Mockito.`when`(provider.facesEnrollMega(anyArg(), anyArg())).thenThrow(
            com.quickpitik.service.ai.AiApiException(org.springframework.http.HttpStatus.SERVICE_UNAVAILABLE, null, "Timed out"))
        runCatching { batchIndexer.drainEvent(event.id, java.time.OffsetDateTime.now().plusMinutes(1)) }
        assertAll(
            { assertNotNull(photos.findById(photo.id).orElseThrow().indexingWorkToken) },
            { assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE photo_id = ? AND finished_at IS NULL AND job_ids ->> 'BIB' = ?",
                Int::class.java, photo.id, bibJob)) },
        )
    }

    @Test fun `batch preparation preserves a flag committed while original bytes load`() {
        org.mockito.Mockito.`when`(storage.getBytes(photo.s3Key)).thenAnswer {
            tx.executeWithoutResult { photos.hideForFlag(photo.id) }
            byteArrayOf(1)
        }
        org.mockito.Mockito.`when`(provider.bibsRecognizeMega(anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(UUID.randomUUID().toString()))
        org.mockito.Mockito.`when`(provider.facesEnrollMega(anyArg(), anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(UUID.randomUUID().toString()))
        assertTrue(batchIndexer.drainEvent(event.id, java.time.OffsetDateTime.now().plusMinutes(1)))
        assertEquals(PhotoStatus.HIDDEN, photos.findById(photo.id).orElseThrow().status)
    }

    @Test fun `V70 preserves already submitted batch writers as unresolved work`() {
        val schema = "fence_upgrade_${UUID.randomUUID().toString().replace("-", "")}"
        fun migrate(target: String) = org.flywaydb.core.Flyway.configure().dataSource(dataSource)
            .locations("classpath:db/migration").defaultSchema(schema).schemas(schema).target(target).load().migrate()
        val oldPhoto = UUID.randomUUID()
        val batch = UUID.randomUUID()
        val eventId = UUID.fromString("11111111-1111-1111-1111-111111111111")
        try {
            migrate("69")
            jdbc.update("INSERT INTO $schema.photos (id,event_id,s3_key,status,price_php,indexing_status) VALUES (?,?,'synthetic/legacy','LIVE',125,'BATCHING')", oldPhoto,eventId)
            jdbc.update("INSERT INTO $schema.ai_index_batches (id,event_id,status,created_at) VALUES (?,?,'SUBMITTED',CURRENT_TIMESTAMP)",batch,eventId)
            jdbc.update("INSERT INTO $schema.ai_index_batch_photos (batch_id,idx,photo_id) VALUES (?,0,?)",batch,oldPhoto)
            jdbc.update("INSERT INTO $schema.ai_index_jobs (id,batch_id,kind,ai_job_id,status,submitted_at) VALUES (?,?,'FACE','legacy-face','SUBMITTED',CURRENT_TIMESTAMP)",UUID.randomUUID(),batch)
            migrate("70")
            assertNotNull(jdbc.queryForObject("SELECT indexing_work_token FROM $schema.photos WHERE id = ?",UUID::class.java,oldPhoto))
            assertEquals(1,jdbc.queryForObject("SELECT count(*) FROM $schema.photo_work_outputs WHERE photo_id = ? AND job_ids ->> 'FACE' = 'legacy-face' AND finished_at IS NULL",Int::class.java,oldPhoto))
        } finally { jdbc.execute("DROP SCHEMA IF EXISTS $schema CASCADE") }
    }

    @Test fun `an older unresolved output still prevents deletion after its photo token changes`() {
        val claim = assertNotNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        // Legacy batches can overlap. Finishing the latest token is not proof
        // that another recorded operation on the same photo ended.
        tx.executeWithoutResult { photos.finishWork(photo.id, claim.token) }
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertEquals(listOf(photo.id), manager.permanentDelete(owner.id, event.id, listOf(photo.id)).failed)
        assertTrue(photos.existsById(photo.id))
    }

    @Test fun `an older unresolved output prevents a replacement claim`() {
        val claim = assertNotNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        tx.executeWithoutResult { photos.finishWork(photo.id, claim.token) }
        assertNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
    }

    @Test fun `restricted and claimed rows do not occupy producer backlog pages`() {
        photo.status = PhotoStatus.PROCESSING
        photos.saveAndFlush(photo)
        val cutoff = java.time.OffsetDateTime.now().plusMinutes(1)
        val page = org.springframework.data.domain.Pageable.unpaged()
        fun assertExcluded() = assertAll(
            { assertFalse(photos.findWatermarkBacklog(5, cutoff, page).any { it.id == photo.id }) },
            { assertFalse(photos.findIndexingBacklog(listOf(IndexingStatus.PENDING), 5, cutoff, page).any { it.id == photo.id }) },
            { assertFalse(event.id in photos.findEventsWithIndexingBacklog(listOf(IndexingStatus.PENDING), 5, cutoff, page)) },
            { assertTrue(photos.findIndexingBacklogForEvent(event.id, listOf(IndexingStatus.PENDING), 5, cutoff, page).isEmpty()) },
        )
        val item = verifiedCase()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertExcluded()
        item.state = RemovalItemState.RETAINED
        items.saveAndFlush(item)
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertNotNull(fence.claim(photo.id, PhotoWorkKind.WATERMARK))
        assertNotNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        assertExcluded()
    }

    @Test fun `completed expired batch retains targets and permits a fresh indexing attempt`() {
        val bibJob = UUID.randomUUID().toString()
        val faceJob = UUID.randomUUID().toString()
        org.mockito.Mockito.`when`(storage.getBytes(photo.s3Key)).thenReturn(byteArrayOf(1))
        org.mockito.Mockito.`when`(provider.bibsRecognizeMega(anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(bibJob))
        org.mockito.Mockito.`when`(provider.facesEnrollMega(anyArg(), anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(faceJob))
        assertTrue(batchIndexer.drainEvent(event.id, java.time.OffsetDateTime.now().plusMinutes(1)))
        jdbc.update("UPDATE photos SET indexing_work_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE id = ?", photo.id)
        org.mockito.Mockito.`when`(provider.jobStatus(faceJob, 0, 1)).thenReturn(com.quickpitik.dto.ai.JobStatusResult(
            job_id = faceJob, status = "completed", result = listOf(mapOf("ref" to photo.id.toString(), "person_id" to "expired-person"))))
        org.mockito.Mockito.`when`(provider.jobStatus(bibJob, 0, 1)).thenReturn(com.quickpitik.dto.ai.JobStatusResult(
            job_id = bibJob, status = "completed", result = emptyList()))
        batchIngest.ingest(faceJob)
        batchIngest.ingest(bibJob)
        val fresh = photos.findById(photo.id).orElseThrow()
        assertAll(
            { assertEquals(IndexingStatus.PENDING, fresh.indexingStatus) },
            { assertTrue(fresh.facePersons.isEmpty()) },
            { assertNull(fresh.indexingWorkToken) },
            { assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE photo_id = ? AND person_ids @> '[\"expired-person\"]'::jsonb AND finished_at IS NOT NULL", Int::class.java, photo.id)) },
        )
    }

    @Test fun `batch locks match PostgreSQL UUID order across the signed boundary`() {
        photos.delete(photo)
        val low = UUID(0L, UUID.randomUUID().leastSignificantBits)
        val high = UUID(Long.MIN_VALUE, UUID.randomUUID().leastSignificantBits)
        listOf(high, low).forEach { id -> photos.saveAndFlush(Photo(id = id, eventId = event.id,
            photographerId = owner.id, s3Key = "synthetic/$id", pricePhp = BigDecimal.ONE)) }
        org.mockito.Mockito.`when`(storage.getBytes(anyArg())).thenReturn(byteArrayOf(1))
        org.mockito.Mockito.`when`(provider.bibsRecognizeMega(anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(UUID.randomUUID().toString()))
        org.mockito.Mockito.`when`(provider.facesEnrollMega(anyArg(), anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(UUID.randomUUID().toString()))
        val first = java.util.concurrent.atomic.AtomicReference<UUID>()
        val begin = java.util.concurrent.CountDownLatch(1)
        val lowHeld = java.util.concurrent.CountDownLatch(1)
        val highHeld = java.util.concurrent.CountDownLatch(1)
        org.mockito.Mockito.doAnswer { invocation ->
            val claim = invocation.getArgument<com.quickpitik.service.photos.PhotoWorkClaim>(0)
            if (first.compareAndSet(null, claim.photoId)) {
                begin.countDown()
                assertTrue(lowHeld.await(5, java.util.concurrent.TimeUnit.SECONDS))
                val result = invocation.callRealMethod()
                if (claim.photoId == high) highHeld.countDown()
                result
            } else invocation.callRealMethod()
        }.`when`(fence).isCurrent(anyArg())
        val executor = java.util.concurrent.Executors.newFixedThreadPool(2)
        try {
            val competing = executor.submit(java.util.concurrent.Callable {
                runCatching { tx.executeWithoutResult {
                    assertTrue(begin.await(5, java.util.concurrent.TimeUnit.SECONDS))
                    photos.findAllByIdForUpdate(listOf(low))
                    lowHeld.countDown()
                    if (first.get() == high) assertTrue(highHeld.await(5, java.util.concurrent.TimeUnit.SECONDS))
                    // Same ascending UUID order used by the shared checkout/removal lock query.
                    photos.findAllByIdForUpdate(listOf(low, high))
                } }.exceptionOrNull()
            })
            val producer = executor.submit(java.util.concurrent.Callable {
                runCatching { batchIndexer.drainEvent(event.id, java.time.OffsetDateTime.now().plusMinutes(1)) }.exceptionOrNull()
            })
            assertAll(
                { assertNull(producer.get(10, java.util.concurrent.TimeUnit.SECONDS), "Batch transaction must not deadlock") },
                { assertNull(competing.get(10, java.util.concurrent.TimeUnit.SECONDS), "Shared-order transaction must not deadlock") },
            )
        } finally { executor.shutdownNow() }
    }

    private fun submitPollingBatch(): Pair<String, String> {
        val bib = UUID.randomUUID().toString()
        val face = UUID.randomUUID().toString()
        org.mockito.Mockito.`when`(storage.getBytes(photo.s3Key)).thenReturn(byteArrayOf(1))
        org.mockito.Mockito.`when`(provider.bibsRecognizeMega(anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(bib))
        org.mockito.Mockito.`when`(provider.facesEnrollMega(anyArg(), anyArg())).thenReturn(com.quickpitik.dto.ai.JobCreateResult(face))
        assertTrue(batchIndexer.drainEvent(event.id, java.time.OffsetDateTime.now().plusMinutes(1)))
        return bib to face
    }

    private fun completePollingBatch(jobs: Pair<String, String>) {
        org.mockito.Mockito.`when`(provider.jobStatus(jobs.second, 0, 1)).thenReturn(com.quickpitik.dto.ai.JobStatusResult(
            job_id = jobs.second, status = "completed", result = listOf(mapOf("ref" to photo.id.toString(), "person_id" to "polled-person"))))
        org.mockito.Mockito.`when`(provider.jobStatus(jobs.first, 0, 1)).thenReturn(com.quickpitik.dto.ai.JobStatusResult(
            job_id = jobs.first, status = "completed", result = emptyList()))
        batchIngest.ingest(jobs.second)
        batchIngest.ingest(jobs.first)
    }

    @Test fun `configured six hundred second first poll can publish a completed batch`() {
        val submitted = clock.instant()
        val jobs = submitPollingBatch()
        org.mockito.Mockito.doReturn(submitted.plusSeconds(601)).`when`(clock).instant()
        completePollingBatch(jobs)
        val fresh = photos.findById(photo.id).orElseThrow()
        assertAll(
            { assertEquals(IndexingStatus.INDEXED, fresh.indexingStatus) },
            { assertEquals(listOf("polled-person"), fresh.facePersons.map { it.aiPersonId }) },
            { assertEquals(1, fresh.indexingAttempts) },
        )
    }

    @Test fun `observed running batch renews bounded validity without another indexing attempt`() {
        val jobs = submitPollingBatch()
        val expiry = photos.findById(photo.id).orElseThrow().indexingWorkExpiresAt!!.toInstant()
        org.mockito.Mockito.doReturn(expiry.minusSeconds(10)).`when`(clock).instant()
        org.mockito.Mockito.`when`(provider.jobStatus(jobs.second, 0, 1)).thenReturn(com.quickpitik.dto.ai.JobStatusResult(
            job_id = jobs.second, status = "processing"))
        batchIngest.ingest(jobs.second)
        val renewedExpiry = photos.findById(photo.id).orElseThrow().indexingWorkExpiresAt!!.toInstant()
        val ledgerMatches = jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs w JOIN photos p " +
            "ON p.indexing_work_token = w.token WHERE p.id = ? AND w.expires_at = p.indexing_work_expires_at " +
            "AND w.finished_at IS NULL", Int::class.java, photo.id)
        org.mockito.Mockito.doReturn(expiry.plusSeconds(10)).`when`(clock).instant()
        completePollingBatch(jobs)
        val fresh = photos.findById(photo.id).orElseThrow()
        assertAll(
            { assertTrue(renewedExpiry.isAfter(expiry), "Observed running work needs another bounded polling window") },
            { assertEquals(1, ledgerMatches, "Renewal must preserve matching unresolved ledger expiry") },
            { assertEquals(IndexingStatus.INDEXED, fresh.indexingStatus) },
            { assertEquals(listOf("polled-person"), fresh.facePersons.map { it.aiPersonId }) },
            { assertEquals(1, fresh.indexingAttempts) },
        )
    }

    @Test fun `ai-api to Rekognition switch retains prior provider targets`() =
        assertProviderProvenance("ai_api", com.quickpitik.config.AiProvider.REKOGNITION)

    @Test fun `Rekognition to ai-api switch retains prior provider targets`() =
        assertProviderProvenance("rekognition", com.quickpitik.config.AiProvider.AI_API)

    @Test fun `legacy missing provider remains explicit uncertainty in prior targets`() =
        assertProviderProvenance(null, com.quickpitik.config.AiProvider.AI_API)

    private fun assertProviderProvenance(priorProvider: String?, nextProvider: com.quickpitik.config.AiProvider) {
        photo.indexedProvider = priorProvider
        photo.facePersons.add(PhotoFacePersonEmbed(0, "prior-person"))
        photos.saveAndFlush(photo)
        val nextFence = PhotoWorkFence(photos, com.quickpitik.config.StorageProperties(), com.quickpitik.config.AiApiProperties(),
            com.quickpitik.config.AiProperties(provider = nextProvider), com.quickpitik.config.RekognitionProperties(),
            com.fasterxml.jackson.databind.ObjectMapper(), java.time.Clock.systemUTC())
        val claim = assertNotNull(tx.execute { nextFence.claim(photo.id, PhotoWorkKind.INDEXING) })
        fence.recordPerson(claim, "new-person")
        tx.executeWithoutResult {
            photos.findById(photo.id).orElseThrow().apply {
                indexedProvider = nextProvider.name.lowercase()
                facePersons.clear()
                facePersons.add(PhotoFacePersonEmbed(0, "new-person"))
            }
        }
        // to_jsonb also gives a meaningful assertion RED before new columns exist.
        val targets = jdbc.queryForMap("SELECT to_jsonb(w) ->> 'prior_provider' AS old_provider, " +
            "to_jsonb(w) -> 'prior_person_ids' ->> 0 AS old_person, provider, person_ids ->> 0 AS new_person, " +
            "jsonb_array_length(person_ids) AS new_count FROM photo_work_outputs w WHERE token = ?", claim.token)
        assertAll(
            { assertEquals(priorProvider, targets["old_provider"]) },
            { assertEquals("prior-person", targets["old_person"]) },
            { assertEquals(nextProvider.name.lowercase(), targets["provider"]) },
            { assertEquals("new-person", targets["new_person"]) },
            { assertEquals(1, targets["new_count"]) },
        )
    }

    @Test fun `face before expiry and bib after expiry preserve actual provider for next claim`() {
        // This provenance fixture requires enabled cleanup and an available old adapter.
        // Keep both local to this test; the SDK mock confirms the old collection is absent.
        val sdk = org.mockito.Mockito.mock(software.amazon.awssdk.services.rekognition.RekognitionClient::class.java)
        org.mockito.Mockito.`when`(sdk.listFaces(anyArg<software.amazon.awssdk.services.rekognition.model.ListFacesRequest>()))
            .thenThrow(software.amazon.awssdk.services.rekognition.model.ResourceNotFoundException.builder().message("Synthetic absent collection").build())
        val oldAdapter = com.quickpitik.service.ai.RekognitionAiClient(sdk, com.quickpitik.config.RekognitionProperties())
        val oldAdapters = org.springframework.beans.factory.support.StaticListableBeanFactory(mapOf("oldAdapter" to oldAdapter))
        val enabledProps = context.getBean(com.quickpitik.config.AiApiProperties::class.java).copy(enabled = true)
        val erasure = com.quickpitik.service.photos.PhotoFaceErasureService(photos, provider,
            oldAdapters.getBeanProvider(com.quickpitik.service.ai.RekognitionAiClient::class.java), enabledProps,
            fence, context.getBean(com.fasterxml.jackson.databind.ObjectMapper::class.java), tx, clock)
        batchIndexer = com.quickpitik.service.photographer.PhotoBatchIndexingService(photos,
            context.getBean(AiIndexBatchRepository::class.java), context.getBean(AiIndexJobRepository::class.java),
            storage, provider, enabledProps, tx, fence, erasure)
        photo.indexedProvider = "rekognition"
        photo.facePersons.add(PhotoFacePersonEmbed(0, "${event.id}.${photo.id}"))
        photos.saveAndFlush(photo)
        val jobs = submitPollingBatch()
        org.mockito.Mockito.`when`(provider.jobStatus(jobs.second, 0, 1)).thenReturn(com.quickpitik.dto.ai.JobStatusResult(
            job_id = jobs.second, status = "completed", result = listOf(mapOf("ref" to photo.id.toString(), "person_id" to "split-person"))))
        batchIngest.ingest(jobs.second)
        val partial = photos.findById(photo.id).orElseThrow()
        org.mockito.Mockito.doReturn(partial.indexingWorkExpiresAt!!.toInstant().plusSeconds(1)).`when`(clock).instant()
        org.mockito.Mockito.`when`(provider.jobStatus(jobs.first, 0, 1)).thenReturn(com.quickpitik.dto.ai.JobStatusResult(
            job_id = jobs.first, status = "completed", result = emptyList()))
        batchIngest.ingest(jobs.first)
        val fresh = photos.findById(photo.id).orElseThrow()
        val next = assertNotNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        val targets = jdbc.queryForMap("SELECT prior_provider, prior_person_ids ->> 0 AS prior_person " +
            "FROM photo_work_outputs WHERE token = ?", next.token)
        assertAll(
            { assertEquals("ai_api", partial.indexedProvider, "Partial results must carry their actual provider") },
            { assertEquals(IndexingStatus.PENDING, fresh.indexingStatus) },
            { assertEquals("ai_api", targets["prior_provider"]) },
            { assertEquals("split-person", targets["prior_person"]) },
            { assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE photo_id = ? " +
                "AND provider = 'ai_api' AND person_ids @> '[\"split-person\"]'::jsonb AND finished_at IS NOT NULL", Int::class.java, photo.id)) },
        )
    }

    @Test fun `ai-api batch keeps actual provider after configuration switches to Rekognition`() {
        val jobs = submitPollingBatch()
        org.mockito.Mockito.doReturn(com.quickpitik.config.AiProvider.REKOGNITION).`when`(aiProperties).provider
        completePollingBatch(jobs)
        val fresh = photos.findById(photo.id).orElseThrow()
        val next = assertNotNull(fence.claim(photo.id, PhotoWorkKind.INDEXING))
        val targets = jdbc.queryForMap("SELECT provider, prior_provider, prior_person_ids ->> 0 AS prior_person " +
            "FROM photo_work_outputs WHERE token = ?", next.token)
        assertAll(
            { assertEquals(IndexingStatus.INDEXED, fresh.indexingStatus) },
            { assertEquals("ai_api", fresh.indexedProvider) },
            { assertEquals("rekognition", targets["provider"]) },
            { assertEquals("ai_api", targets["prior_provider"]) },
            { assertEquals("polled-person", targets["prior_person"]) },
        )
    }
}
