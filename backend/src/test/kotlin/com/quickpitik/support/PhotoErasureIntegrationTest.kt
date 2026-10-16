package com.quickpitik.support

import com.quickpitik.dto.photos.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photos.*
import com.quickpitik.service.photographer.PhotographerPhotoService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.mock.mockito.MockBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.*

class PhotoErasureIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var coverage: EventPhotographerRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var eraser: PhotoErasureService
    @Autowired lateinit var manager: PhotographerPhotoService
    @Autowired lateinit var availability: PhotoAvailabilityService
    @Autowired lateinit var fence: PhotoWorkFence
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var jobs: PhotoErasureJobRepository
    @Autowired lateinit var intents: PhotoUploadIntentRepository
    @Autowired lateinit var jdbc: org.springframework.jdbc.core.JdbcTemplate
    @Autowired lateinit var sweep: PhotoErasureSweep
    @Autowired lateinit var faceErasure: PhotoFaceErasureService
    @Autowired lateinit var metadata: PhotoMetadataRepository
    @Autowired lateinit var grants: DownloadGrantRepository
    @Autowired lateinit var json: com.fasterxml.jackson.databind.ObjectMapper
    @Autowired lateinit var uploads: com.quickpitik.service.photographer.PhotoUploadService
    @Autowired lateinit var settings: PhotographerSettingsRepository
    @Autowired lateinit var registry: io.micrometer.core.instrument.MeterRegistry
    @MockBean lateinit var storage: StorageService
    lateinit var owner: User
    lateinit var admin: User
    lateinit var event: Event
    lateinit var photo: Photo
    private val blobs = mutableSetOf<String>()
    private var failingKey: String? = null
    private var crash = false
    private var afterDelete: (() -> Unit)? = null

    @BeforeEach fun seed() {
        owner = users.saveAndFlush(User(email = "eraser-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Owner", role = Role.PHOTOGRAPHER))
        admin = users.saveAndFlush(User(email = "eraser-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Admin", role = Role.ADMIN))
        event = events.saveAndFlush(Event(slug = "eraser-${UUID.randomUUID()}", name = "Erasure", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE, photoCount = 1))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "erasure/${UUID.randomUUID()}/original", thumbnailS3Key = "erasure/${UUID.randomUUID()}/thumb",
            watermarkS3Key = "erasure/${UUID.randomUUID()}/watermark", pricePhp = BigDecimal("125")))
        coverage.saveAndFlush(EventPhotographer(EventPhotographerId(event.id, owner.id), photoCount = 1, firstUploadAt = OffsetDateTime.now()))
        blobs.clear()
        blobs.addAll(listOf(photo.s3Key, photo.thumbnailS3Key!!, photo.watermarkS3Key!!))
        failingKey = null
        crash = false
        afterDelete = null
        Mockito.`when`(storage.exists(Mockito.anyString())).thenAnswer { it.getArgument<String>(0) in blobs }
        Mockito.doAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive(), "Storage I/O holds a database transaction")
            val key = it.getArgument<String>(0)
            if (key == failingKey) throw java.io.IOException("secret provider URL must not escape")
            blobs.remove(key)
            afterDelete?.let { callback -> afterDelete = null; callback() }
            if (crash) { crash = false; throw AssertionError("simulated process death after storage delete") }
            null
        }.`when`(storage).delete(Mockito.anyString())
    }

    private fun privacy(): PhotoErasureCommand {
        val request = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id,
            eventNameSnapshot = event.name, requesterContact = "synthetic", supportReference = "test",
            requesterReason = "test", publicToOwnerReason = "test", creatorId = admin.id, creationCommandId = UUID.randomUUID()))
        val item = items.saveAndFlush(PhotoRemovalItem(requestId = request.id, photoIdSnapshot = photo.id,
            photoId = photo.id, ownerIdSnapshot = owner.id, matchSource = RemovalMatchSource.MANUAL,
            state = RemovalItemState.REMOVAL_AUTHORIZED, adminDecision = RemovalDecision.REMOVE,
            adminDecisionActorId = admin.id, adminDecisionReason = "confirmed privacy decision"))
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, admin.id) }
        return PhotoErasureCommand(photo.id, PhotoErasurePurpose.PRIVACY_REMOVAL, RemovalActorType.ADMIN,
            admin.id, "confirmed privacy decision", request.id, item.id, UUID.randomUUID())
    }

    @Test fun bothCallersUseOneEraser() {
        val command = privacy()
        val firstJobId = eraser.request(command)
        assertEquals(firstJobId, eraser.request(command.copy(commandId = UUID.randomUUID())))
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(firstJobId))

        val ordinary = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "erasure/${UUID.randomUUID()}/ordinary", pricePhp = BigDecimal("125"), deletedAt = OffsetDateTime.now()))
        val job = eraser.request(PhotoErasureCommand(ordinary.id, PhotoErasurePurpose.PHOTOGRAPHER_DELETE,
            RemovalActorType.PHOTOGRAPHER, owner.id, "Photographer permanent deletion", commandId = UUID.randomUUID()))
        assertEquals(listOf(ordinary.id), manager.permanentDelete(owner.id, event.id, listOf(ordinary.id)).deleted)
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
    }

    @Test fun soldPrivacyRemovalPreservesTombstoneAndLedger() {
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "synthetic@test.local",
            paymentMethodWire = "gcash", status = OrderStatus.FULFILLED, totalPhp = photo.pricePhp))
        val financialId = OrderItemId(order.id, photo.id)
        orderItems.saveAndFlush(OrderItem(financialId, pricePhpAtPurchase = photo.pricePhp))
        val before = orderItems.findAll().map { it.id }.toSet()
        val job = eraser.request(privacy())
        val completed = registry.counter("qp.removal.erasure", "outcome", "COMPLETE").count()
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertEquals(completed + 1.0, registry.counter("qp.removal.erasure", "outcome", "COMPLETE").count())
        val tombstone = photos.findById(photo.id).orElseThrow()
        assertNotNull(tombstone.purgedAt)
        assertTrue(tombstone.s3Key.isBlank())
        assertEquals(before, orderItems.findAll().map { it.id }.toSet())
        assertTrue(blobs.isEmpty())
    }

    @Test fun failedDerivativeRetriesWithoutDoubleCounting() {
        val job = eraser.request(privacy())
        failingKey = photo.thumbnailS3Key
        assertEquals(PhotoErasureOutcome.RETRY_REQUIRED, eraser.attempt(job))
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
        assertFalse(photo.s3Key in blobs)
        failingKey = null
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertEquals(1, 1 - events.findById(event.id).orElseThrow().photoCount)
        assertEquals(0, coverage.findById(EventPhotographerId(event.id, owner.id)).orElseThrow().photoCount)
    }

    @Test fun crashAfterDeleteBeforeSettleIsIdempotent() {
        val job = eraser.request(privacy())
        crash = true
        assertFailsWith<AssertionError> { eraser.attempt(job) }
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        jdbc.update("UPDATE photo_erasure_jobs SET lease_until = CURRENT_TIMESTAMP - INTERVAL '1 minute' WHERE id = ?", job)
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertTrue(blobs.isEmpty())
        assertEquals(0, events.findById(event.id).orElseThrow().photoCount)
    }

    @Test fun lateWriterPreventsFalseCompletion() {
        val writer = fence.claim(photo.id, PhotoWorkKind.WATERMARK)!!
        val job = eraser.request(privacy())
        assertNotNull(photos.findById(photo.id).orElseThrow().deletionStartedAt)
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
        val lateKey = "events/${event.id}/photos/${photo.id}/watermark.jpg"
        blobs.add(lateKey)
        assertFalse(fence.isCurrent(writer))
        fence.finish(writer)
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertFalse(lateKey in blobs)
    }

    @Test fun `known verifying upload keeps blobs intact even after its lease expires`() {
        val now = OffsetDateTime.now()
        intents.saveAndFlush(PhotoUploadIntent(photographerId = owner.id, eventId = event.id,
            photoId = photo.id, contentHash = "d".repeat(64), contentType = "image/jpeg", sizeBytes = 1,
            stagingKey = "staging/${photo.id}", originalKey = photo.s3Key, state = PhotoUploadIntentState.VERIFYING,
            leaseToken = UUID.randomUUID(), leaseUntil = now.minusMinutes(1), createdAt = now.minusDays(10),
            putExpiresAt = now.minusDays(9), expiresAt = now.minusDays(8), cleanedAt = now.minusDays(8)))
        val before = blobs.toSet()
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        assertEquals(before, blobs, "An unresolved upload can still be copying the original")
    }

    @Test fun `surviving manifest cleans objects after live photo row disappears`() {
        val command = privacy()
        val job = eraser.request(command)
        tx.executeWithoutResult { photos.deleteById(photo.id) }
        assertTrue(blobs.isNotEmpty())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertTrue(blobs.isEmpty())
        val saved = jobs.findById(job).orElseThrow()
        assertEquals(command.authorizedActorId, saved.authorizedActorId)
        assertEquals(RemovalActorType.SYSTEM, saved.executionActorType)
        assertNull(saved.executionActorId)
        assertTrue(saved.resourceOutcomes.values.all { it == "ABSENT" })
        assertNotNull(saved.completedAt)
    }

    @Test fun `shared derivative is protected and prevents a completion claim`() {
        photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = photo.thumbnailS3Key!!, pricePhp = BigDecimal("125")))
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.RETRY_REQUIRED, eraser.attempt(job))
        assertTrue(photo.thumbnailS3Key in blobs)
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
        assertEquals("SHARED_REFERENCE", jobs.findById(job).orElseThrow().resourceOutcomes[photo.thumbnailS3Key])
    }

    @Test fun `new output reopens a completed job after ordinary row deletion`() {
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertEquals(listOf(photo.id), manager.permanentDelete(owner.id, event.id, listOf(photo.id)).deleted)
        val job = jobs.findByPhotoId(photo.id)!!
        val key = "events/${event.id}/photos/${photo.id}/watermark.jpg"
        val token = UUID.randomUUID()
        blobs.add(key)
        tx.executeWithoutResult {
            photos.recordWork(token, photo.id, event.id, 0, "WATERMARK", "storage", OffsetDateTime.now(), key, null, null, "[]")
        }
        assertEquals(PhotoErasureOutcome.PENDING, jobs.findById(job.id).orElseThrow().outcome)
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job.id))
        tx.executeWithoutResult { photos.finishWorkOutput(token, OffsetDateTime.now()) }
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job.id))
        assertFalse(key in blobs)
        assertFalse(photos.existsById(photo.id))
    }

    @Test fun `due ordering retries later jobs while a failed job backs off`() {
        val first = eraser.request(privacy())
        failingKey = photo.thumbnailS3Key
        assertEquals(PhotoErasureOutcome.RETRY_REQUIRED, eraser.attempt(first))
        val ordinary = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "erasure/${UUID.randomUUID()}/due", pricePhp = BigDecimal("125"), deletedAt = OffsetDateTime.now()))
        val second = eraser.request(PhotoErasureCommand(ordinary.id, PhotoErasurePurpose.PHOTOGRAPHER_DELETE,
            RemovalActorType.PHOTOGRAPHER, owner.id, "Photographer permanent deletion", commandId = UUID.randomUUID()))
        val due = jobs.findDue(java.time.Instant.now().plusSeconds(1), org.springframework.data.domain.PageRequest.of(0, 100))
        assertFalse(first in due)
        assertTrue(second in due)
        assertFalse(jobs.findById(first).orElseThrow().lastError!!.contains("secret"))
    }

    @Test fun `sweep resumes a persisted due job without another user request`() {
        val job = eraser.request(privacy())
        assertTrue(sweep.sweepDue(java.time.Instant.now().plusSeconds(1), 100) > 0)
        assertEquals(PhotoErasureOutcome.COMPLETE, jobs.findById(job).orElseThrow().outcome)
        assertTrue(blobs.isEmpty())
    }

    @Test fun `failed orphan face cleanup moves behind later due targets`() {
        val removedId = UUID.randomUUID()
        val token = UUID.randomUUID()
        tx.executeWithoutResult {
            photos.recordWork(token, removedId, event.id, 0, "INDEXING", "ai_api", OffsetDateTime.now(), null,
                removedId.toString(), null, "[]")
            photos.finishWorkOutput(token, OffsetDateTime.now())
        }
        val page = org.springframework.data.domain.PageRequest.of(0, 100)
        assertTrue(photos.faceWorkPending(page).any { it.photoId == removedId })
        val reaper = com.quickpitik.service.photographer.PhotoOrphanReaper(photos,
            Mockito.mock(com.quickpitik.service.ai.AiApiClient::class.java), com.quickpitik.config.AiApiProperties(enabled = true), faceErasure,
            com.quickpitik.config.PhotoRemovalProperties(), java.time.Clock.systemUTC())
        reaper.reap()
        assertFalse(photos.faceWorkPending(page).any { it.photoId == removedId }, "Persistent failures monopolize the first page")
    }

    @Test fun `privacy authorization joins an ordinary claimed job without losing its tombstone`() {
        manager.trash(owner.id, event.id, listOf(photo.id))
        val ordinary = PhotoErasureCommand(photo.id, PhotoErasurePurpose.PHOTOGRAPHER_DELETE,
            RemovalActorType.PHOTOGRAPHER, owner.id, "Photographer permanent deletion", commandId = UUID.randomUUID())
        val job = eraser.request(ordinary)
        afterDelete = {
            val authorized = privacy()
            val result = runCatching { eraser.request(authorized) }
            assertTrue(result.isSuccess, "An authorized privacy request must join the existing cleanup job")
            assertEquals(job, result.getOrThrow())
        }
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        assertTrue(photos.existsById(photo.id), "Stale ordinary settlement deleted the newly required tombstone")
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertNotNull(photos.findById(photo.id).orElseThrow().purgedAt)
        assertEquals(owner.id, jobs.findById(job).orElseThrow().authorizedActorId)
        assertEquals(admin.id, jobs.findById(job).orElseThrow().authorizations.single().actorId)
        assertEquals(1, jobs.findAll().count { it.photoId == photo.id })
    }

    @Test fun `late target recording follows the photo then erasure job lock order`() {
        val writer = fence.claim(photo.id, PhotoWorkKind.WATERMARK)!!
        fence.finish(writer)
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertNotNull(photos.findById(photo.id).orElseThrow().purgedAt)
        val executor = java.util.concurrent.Executors.newSingleThreadExecutor()
        var callback: java.util.concurrent.Future<*>? = null
        try {
            val settlement = runCatching {
                tx.executeWithoutResult {
                    photos.findAllByIdForUpdate(listOf(photo.id))
                    callback = executor.submit { fence.recordJob(writer, "LATE", "late-job") }
                    val deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(5)
                    while (System.nanoTime() < deadline && jdbc.queryForObject(
                            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'",
                            Int::class.java) == 0) Thread.sleep(10)
                    jobs.lock(job)
                }
            }
            val producer = runCatching { callback!!.get(10, java.util.concurrent.TimeUnit.SECONDS) }
            assertTrue(settlement.isSuccess && producer.isSuccess, "Late target recording deadlocked the erasure settlement")
        } finally { executor.shutdownNow() }
    }

    @Test fun `face cleanup is independent from blob absence and metadata is minimized only after confirmation`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, "recorded-person", FaceProvider.AI_API))
        photo.bibs.add(PhotoBibEmbed("42", BigDecimal("0.9")))
        photo.indexingAttempts = 1
        photo.indexedProvider = "ai_api"
        photos.saveAndFlush(photo)
        metadata.saveAndFlush(PhotoMetadata(photo.id, originalFilename = "private-name.jpg"))
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.RETRY_REQUIRED, eraser.attempt(job))
        assertTrue(blobs.isEmpty())
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
        assertTrue(metadata.existsById(photo.id))
        assertNotEquals("DELETED", jobs.findById(job).orElseThrow().faceOutcome)

        val ai = Mockito.mock(com.quickpitik.service.ai.AiApiClient::class.java)
        Mockito.`when`(ai.listPersonsForEvent(event.id)).thenReturn(listOf(
            com.quickpitik.service.ai.AiPersonRef("recorded-person", OffsetDateTime.now(), photo.id.toString())))
        Mockito.`when`(ai.erasePhotoFaces(event.id, "recorded-person")).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            com.quickpitik.dto.ai.FaceErasureResult.DELETED
        }
        @Suppress("UNCHECKED_CAST")
        val rekognition = Mockito.mock(org.springframework.beans.factory.ObjectProvider::class.java) as
            org.springframework.beans.factory.ObjectProvider<com.quickpitik.service.ai.RekognitionAiClient>
        val faceService = PhotoFaceErasureService(photos, ai, rekognition,
            com.quickpitik.config.AiApiProperties(enabled = true), fence, json, tx, java.time.Clock.systemUTC())
        val retry = PhotoErasureService(photos, jobs, intents, items, orderItems, grants, coverage, events, metadata,
            storage, faceService, com.quickpitik.config.PhotoRemovalProperties(), com.quickpitik.config.StorageProperties(),
            tx, json, java.time.Clock.systemUTC(), registry)
        assertEquals(PhotoErasureOutcome.COMPLETE, retry.attempt(job))
        assertFalse(metadata.existsById(photo.id))
        val tombstone = photos.findById(photo.id).orElseThrow()
        assertNotNull(tombstone.purgedAt)
        assertTrue(tombstone.bibs.isEmpty())
        assertTrue(tombstone.facePersons.isEmpty())
        assertTrue(photos.faceWorkForPhoto(event.id, photo.id).all { it.facesErasedAt != null })
    }

    @Test fun `existing future staging recheck is retained even after the original cleanup horizon`() {
        val now = OffsetDateTime.now()
        intents.saveAndFlush(PhotoUploadIntent(photographerId = owner.id, eventId = event.id,
            contentHash = "e".repeat(64), contentType = "image/jpeg", sizeBytes = 1,
            stagingKey = "staging/${photo.id}", originalKey = photo.s3Key, state = PhotoUploadIntentState.COMMITTED,
            resultPhotoId = photo.id, createdAt = now.minusDays(10), putExpiresAt = now.minusDays(9),
            expiresAt = now.minusDays(8), cleanedAt = now.minusDays(8), cleanupAfter = now.plusHours(1)))
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
    }

    @Test fun `one bulk authorization command can request erasure for two photos`() {
        val first = privacy()
        val firstJob = eraser.request(first)
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "erasure/${UUID.randomUUID()}/second", pricePhp = BigDecimal("125")))
        val item = items.saveAndFlush(PhotoRemovalItem(requestId = first.requestId!!, photoIdSnapshot = photo.id,
            photoId = photo.id, ownerIdSnapshot = owner.id, matchSource = RemovalMatchSource.MANUAL,
            state = RemovalItemState.REMOVAL_AUTHORIZED, adminDecision = RemovalDecision.REMOVE,
            adminDecisionActorId = admin.id, adminDecisionReason = "confirmed privacy decision"))
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, admin.id) }
        val second = first.copy(photoId = photo.id, itemId = item.id)
        val result = runCatching { eraser.request(second) }
        assertTrue(result.isSuccess, "One bulk action must retain its command ID across authorized photos")
        assertNotEquals(firstJob, result.getOrThrow())
        assertEquals(first.commandId, jobs.findById(result.getOrThrow()).orElseThrow().commandId)
    }

    @Test fun `staging recheck scheduled during erasure prevents settlement`() {
        val now = OffsetDateTime.now()
        val receipt = intents.saveAndFlush(PhotoUploadIntent(photographerId = owner.id, eventId = event.id,
            contentHash = "f".repeat(64), contentType = "image/jpeg", sizeBytes = 1,
            stagingKey = "staging/${photo.id}", originalKey = photo.s3Key, state = PhotoUploadIntentState.COMMITTED,
            resultPhotoId = photo.id, createdAt = now.minusDays(10), putExpiresAt = now.minusDays(9),
            expiresAt = now.minusDays(8), cleanedAt = now.minusDays(8)))
        val job = eraser.request(privacy())
        afterDelete = { jdbc.update("UPDATE photo_upload_intents SET cleanup_after = ? WHERE photo_id = ?",
            now.plusHours(1), receipt.photoId) }
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
    }

    @Test fun `privacy tombstone retains the exact byte hash to prevent reupload`() {
        val hash = "a".repeat(64)
        photo.contentHash = hash
        photos.saveAndFlush(photo)
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertEquals(hash, photos.findById(photo.id).orElseThrow().contentHash)
        assertNotNull(photos.findFirstByPhotographerIdAndContentHash(owner.id, hash))
    }

    private fun duplicateReceipt(): PhotoUploadIntent {
        settings.saveAndFlush(PhotographerSettings(userId = owner.id, verificationStatus = VerificationStatus.APPROVED))
        event.watermarkPolicy = WatermarkPolicy.NONE
        events.saveAndFlush(event)
        val bytes = javaClass.getResourceAsStream("/photo-metadata/no-tags.jpg")!!.use { it.readAllBytes() }
        val hash = java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes))
        val id = UUID.randomUUID()
        val now = OffsetDateTime.now()
        val receipt = intents.saveAndFlush(PhotoUploadIntent(photoId = id, photographerId = owner.id, eventId = event.id,
            contentHash = hash, contentType = "image/jpeg", sizeBytes = bytes.size.toLong(),
            stagingKey = "staging/duplicate/$id", originalKey = "events/${event.id}/photos/$id/original.jpg",
            putExpiresAt = now.plusMinutes(15), expiresAt = now.plusDays(1)))
        // The pending direct slot predates the competing marketplace insert.
        photo.contentHash = hash
        photos.saveAndFlush(photo)
        blobs.add(receipt.stagingKey)
        val original = com.quickpitik.service.storage.StoredObject(receipt.originalKey, bytes.size.toLong(), "image/jpeg", "etag")
        Mockito.`when`(storage.metadata(receipt.originalKey)).thenAnswer { if (receipt.originalKey in blobs) original else null }
        Mockito.`when`(storage.metadata(receipt.stagingKey)).thenReturn(original.copy(key = receipt.stagingKey))
        Mockito.`when`(storage.copyIfAbsent(receipt.stagingKey, receipt.originalKey, "etag", "image/jpeg")).thenAnswer {
            blobs.add(receipt.originalKey)
            original
        }
        Mockito.`when`(storage.open(receipt.originalKey)).thenAnswer { bytes.inputStream() }
        Mockito.`when`(storage.presignedGetUrl(receipt.originalKey, com.quickpitik.config.StorageProperties().presignedTtl.thumbnail))
            .thenReturn("https://storage.invalid/duplicate")
        return receipt
    }

    private fun commitDuplicate(receipt: PhotoUploadIntent) {
        val failure = runCatching {
            uploads.commitDirectUpload(owner.id, event.id, receipt.photoId, receipt.stagingKey,
                receipt.contentHash, receipt.sizeBytes, receipt.contentType)
        }.exceptionOrNull()
        assertTrue(failure is com.quickpitik.exception.NotFoundException ||
            failure is com.quickpitik.exception.ApiException && failure.code == "UPLOAD_IN_PROGRESS",
            "A removed duplicate must not be returned to its uploader: $failure")
        assertEquals("COMMITTED", jdbc.queryForObject("SELECT state FROM photo_upload_intents WHERE photo_id = ?", String::class.java, receipt.photoId))
        assertEquals(photo.id, jdbc.queryForObject("SELECT result_photo_id FROM photo_upload_intents WHERE photo_id = ?", UUID::class.java, receipt.photoId),
            "The real duplicate INSERT fallback must associate the receipt")
    }

    @Test fun `late duplicate receipt during erasure invalidates the old attempt`() {
        val receipt = duplicateReceipt()
        val job = eraser.request(privacy())
        val revision = jobs.findById(job).orElseThrow().revision
        afterDelete = { commitDuplicate(receipt) }
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        assertTrue(jobs.findById(job).orElseThrow().revision > revision)
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
    }

    @Test fun `late duplicate receipt reopens completed erasure until its copies and obligations are cleared`() {
        val receipt = duplicateReceipt()
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        commitDuplicate(receipt)
        assertEquals(PhotoErasureOutcome.PENDING, jobs.findById(job).orElseThrow().outcome)
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
        assertTrue(job in jobs.findDue(java.time.Instant.now().plusSeconds(1), org.springframework.data.domain.PageRequest.of(0, 100)))
        assertEquals(PhotoErasureOutcome.PENDING, eraser.attempt(job))
        assertFalse(receipt.stagingKey in blobs)
        assertFalse(receipt.originalKey in blobs)
        // Durable late targets still drive cleanup after their live FK disappears.
        tx.executeWithoutResult { photos.deleteById(photo.id) }
        blobs.addAll(listOf(receipt.stagingKey, receipt.originalKey))
        val later = java.time.Clock.offset(java.time.Clock.systemUTC(), java.time.Duration.ofDays(10))
        val resumed = PhotoErasureService(photos, jobs, intents, items, orderItems, grants, coverage, events, metadata,
            storage, faceErasure, com.quickpitik.config.PhotoRemovalProperties(), com.quickpitik.config.StorageProperties(),
            tx, json, later, registry)
        assertEquals(PhotoErasureOutcome.COMPLETE, resumed.attempt(job))
        assertFalse(receipt.stagingKey in blobs)
        assertFalse(receipt.originalKey in blobs)
    }

    @Test fun `late duplicate association follows settlement photo job receipt locks`() {
        val receipt = duplicateReceipt()
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        val executor = java.util.concurrent.Executors.newSingleThreadExecutor()
        var callback: java.util.concurrent.Future<*>? = null
        var producerWaitedForPhoto = false
        try {
            val settlement = runCatching {
                tx.executeWithoutResult {
                    photos.findAllByIdForUpdate(listOf(photo.id))
                    callback = executor.submit { commitDuplicate(receipt) }
                    val deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(5)
                    while (System.nanoTime() < deadline && jdbc.queryForObject(
                            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'",
                            Int::class.java) == 0) Thread.sleep(10)
                    producerWaitedForPhoto = !callback!!.isDone
                    jobs.lock(job)
                    jdbc.queryForObject("SELECT photo_id FROM photo_upload_intents WHERE photo_id = ? FOR UPDATE", UUID::class.java, receipt.photoId)
                }
            }
            val producer = runCatching { callback!!.get(10, java.util.concurrent.TimeUnit.SECONDS) }
            assertTrue(producerWaitedForPhoto, "Duplicate association must wait for the settlement photo lock")
            assertTrue(settlement.isSuccess && producer.isSuccess, "Duplicate association deadlocked erasure settlement")
            assertEquals(PhotoErasureOutcome.PENDING, jobs.findById(job).orElseThrow().outcome)
        } finally { executor.shutdownNow() }
    }
}
