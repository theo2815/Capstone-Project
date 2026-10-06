package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.removal.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import com.quickpitik.service.image.ImagePixelGuard
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.StoredObject
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.*
import org.mockito.Mockito.*
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.transaction.support.TransactionSynchronization
import org.springframework.transaction.IllegalTransactionStateException
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.net.URI
import java.net.http.*
import java.math.BigDecimal
import java.nio.file.Files
import java.nio.file.Path
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32
import java.time.LocalDate
import java.time.Instant
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.*

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.photo-removal.cleanup-interval=PT24H"])
class PhotoRemovalAdminIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var service: PhotoRemovalService
    @Autowired lateinit var evidence: PhotoRemovalEvidenceService
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @MockitoSpyBean lateinit var commands: PhotoRemovalCommandRepository
    @Autowired lateinit var history: PhotoRemovalHistoryRepository
    @Autowired lateinit var selfies: UserSelfieRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var availability: PhotoAvailabilityService
    @Autowired lateinit var reviews: PhotoRemovalReviewRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var tokens: JwtTokenProvider
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var tx: TransactionTemplate
    @MockitoBean lateinit var storage: StorageService
    @LocalServerPort var port = 0
    private val client = HttpClient.newHttpClient()
    private lateinit var admin: User
    private lateinit var event: Event

    @BeforeEach fun fixture() {
        admin = user(Role.ADMIN)
        event = events.saveAndFlush(Event(slug = "intake-${UUID.randomUUID()}", name = "Synthetic Cebu run",
            date = LocalDate.of(2026, 10, 6), location = "Cebu", status = EventStatus.ACTIVE))
        `when`(storage.put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive(), "PUT must run outside transaction")
            StoredObject(it.getArgument(0), (it.getArgument<ByteArray>(1)).size.toLong(), it.getArgument(2))
        }
        doAnswer { assertFalse(TransactionSynchronizationManager.isActualTransactionActive(), "DELETE must run outside transaction"); null }
            .`when`(storage).delete(anyString())
    }

    @Test fun `runnerAndPhotographerCannotReadCaseEvidence`() {
        val case = create()
        val command = UUID.randomUUID()
        evidence.replace(admin.id, case.id, command, case.version, image())
        val key = "photo-removal/${case.id}/evidence/$command"
        val path = Path.of("build/test-storage", key)
        Files.createDirectories(path.parent)
        Files.write(path, image().bytes)
        `when`(storage.open(key)).thenAnswer { image().bytes.inputStream() }
        try {
            val route = "/api/v1/admin/photo-removal-requests/${case.id}/evidence"
            val photographerEvidenceGet = get(route, user(Role.PHOTOGRAPHER))
            assertEquals(403, photographerEvidenceGet.status)
            assertEquals(403, get(route, user(Role.RUNNER)).status)
            assertEquals(401, get(route).status)
            val anonymousDirectStorageReadOfKnownEvidenceKey = get("/storage/$key")
            assertTrue(anonymousDirectStorageReadOfKnownEvidenceKey.status in setOf(401, 403, 404))
            assertTrue(get("/storage/$key", admin).status in setOf(401, 403, 404))
            `when`(storage.open(key)).thenAnswer { image().bytes.inputStream() }
            val authorized = get(route, admin)
            assertEquals(200, authorized.status)
            assertTrue("no-store" in authorized.response.headers().firstValue("Cache-Control").orElse(""))
            assertContentEquals(image().bytes, authorized.response.body())
        } finally { Files.deleteIfExists(path) }
    }

    @Test fun `adminEvidenceStreamCompletes`() {
        val case = create()
        val command = UUID.randomUUID()
        evidence.replace(admin.id, case.id, command, case.version, image())
        val key = "photo-removal/${case.id}/evidence/$command"
        `when`(storage.open(key)).thenAnswer { image().bytes.inputStream() }
        val result = get("/api/v1/admin/photo-removal-requests/${case.id}/evidence", admin)
        assertEquals(200, result.status)
        assertTrue("no-store" in result.response.headers().firstValue("Cache-Control").orElse(""))
        assertContentEquals(image().bytes, result.response.body())
    }

    @Test fun `runner and photographer cannot read private case detail`() {
        val case = create()
        val path = "/api/v1/admin/photo-removal-requests/${case.id}"
        assertEquals(403, get(path, user(Role.RUNNER)).status)
        assertEquals(403, get(path, user(Role.PHOTOGRAPHER)).status)
    }

    @Test fun `evidenceNeverEntersAdminSelfieLibrary`() {
        val case = create()
        evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, image())
        val userSelfiesCreatedForAdmin = selfies.countByUserId(admin.id).toInt()
        assertEquals(0, userSelfiesCreatedForAdmin)
        val body = mapper.writeValueAsString(queries.adminDetail(admin.id, case.id))
        assertFalse(body.contains("photo-removal/"))
        assertFalse(body.contains("dataUrl"))
    }

    @Test fun `create and replacement replay original results and reject changed actor payload or version`() {
        val command = intake()
        val created = service.create(admin.id, command)
        val replacement = UUID.randomUUID()
        val replaced = evidence.replace(admin.id, created.id, replacement, created.version, image())
        assertEquals(created, service.create(admin.id, command))
        assertEquals(replaced, evidence.replace(admin.id, created.id, replacement, created.version, image()))
        assertEquals(409, assertFailsWith<ApiException> { service.create(user(Role.ADMIN).id, command) }.status.value())
        assertEquals(409, assertFailsWith<ApiException> { service.create(admin.id, command.copy(reason = "changed")) }.status.value())
        assertEquals(409, assertFailsWith<ApiException> {
            evidence.replace(admin.id, created.id, replacement, replaced.version, image())
        }.status.value())
        assertEquals(409, assertFailsWith<ApiException> {
            evidence.replace(admin.id, created.id, UUID.randomUUID(), created.version, image())
        }.status.value())
        assertEquals(1, history.findByRequestIdAndCommandId(created.id, replacement).size)
    }

    @Test fun `intake records private support notes without claiming later verification`() {
        val case = service.create(admin.id, intake().copy(adminNote = "Support verified contact; requester acts for self"))
        assertEquals("Support verified contact; requester acts for self", case.verificationNote)
        assertNull(case.verifierId)
        assertNull(case.verifiedAt)
        assertNull(case.verificationMethod)
        assertEquals(404, assertFailsWith<ApiException> { service.create(admin.id, intake().copy(eventId = UUID.randomUUID())) }.status.value())
        assertEquals(400, assertFailsWith<ApiException> { service.create(admin.id, intake().copy(requesterUserId = user(Role.PHOTOGRAPHER).id)) }.status.value())
        assertEquals(400, assertFailsWith<ApiException> { service.create(admin.id, intake().copy(supportReference = "x".repeat(1001))) }.status.value())
    }

    @Test fun `cancellation closes only after confirmed evidence deletion and replays its original result`() {
        val created = create()
        val replaced = evidence.replace(admin.id, created.id, UUID.randomUUID(), created.version, image())
        val key = requests.findById(created.id).orElseThrow().evidenceKey!!
        doThrow(IllegalStateException("synthetic delete failure")).`when`(storage).delete(key)
        val command = CancelRemovalCommand(UUID.randomUUID(), replaced.version, "Support withdrew request")
        val cancelled = service.cancel(admin.id, created.id, command)
        assertEquals(RemovalRequestStatus.RESOLVING, cancelled.status)
        assertNull(cancelled.resolvedAt)
        assertEquals(EvidenceCleanupResult.RETRY_REQUIRED, evidence.purgeResolvedEvidence(created.id))
        doNothing().`when`(storage).delete(key)
        assertEquals(EvidenceCleanupResult.DELETED, evidence.purgeResolvedEvidence(created.id))
        val after = requests.findById(created.id).orElseThrow()
        assertEquals(RemovalRequestStatus.CANCELLED, after.status)
        assertNotNull(after.resolvedAt)
        assertNull(after.evidenceKey)
        assertEquals(cancelled, service.cancel(admin.id, created.id, command))
    }

    @Test fun `ambiguous PUT cannot be cleared by successful deletion or elapsed time`() {
        val case = create()
        doThrow(IllegalStateException("synthetic ambiguous PUT")).`when`(storage).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
        assertFailsWith<ApiException> { evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, image()) }
        val current = queries.adminDetail(admin.id, case.id)
        service.cancel(admin.id, case.id, CancelRemovalCommand(UUID.randomUUID(), current.version, "Withdraw"))
        assertEquals(EvidenceCleanupResult.RETRY_REQUIRED, evidence.purgeResolvedEvidence(case.id))
        assertNull(requests.findById(case.id).orElseThrow().resolvedAt)
        assertTrue(requests.findById(case.id).orElseThrow().outstandingEvidenceKeys.isNotEmpty())
    }

    @Test fun `live photo cancellation persists cleanup and preserves another case restriction`() {
        val created = create()
        val replaced = evidence.replace(admin.id, created.id, UUID.randomUUID(), created.version, image())
        val key = requests.findById(created.id).orElseThrow().evidenceKey!!
        val owner = user(Role.PHOTOGRAPHER)
        val photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("125")))
        val other = create()
        val selected = items.saveAndFlush(PhotoRemovalItem(requestId = created.id, photoIdSnapshot = photo.id,
            photoId = photo.id, ownerIdSnapshot = owner.id, state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL))
        val overlapping = items.saveAndFlush(PhotoRemovalItem(requestId = other.id, photoIdSnapshot = photo.id,
            photoId = photo.id, ownerIdSnapshot = owner.id, state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL))
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertEquals(RemovalState.REVIEW, photos.findById(photo.id).orElseThrow().removalState)
        doThrow(IllegalStateException("synthetic delete failure")).`when`(storage).delete(key)
        val command = CancelRemovalCommand(UUID.randomUUID(), replaced.version, "Withdraw live selected photo")
        val original = service.cancel(admin.id, created.id, command)
        assertEquals(RemovalRequestStatus.RESOLVING, requests.findById(created.id).orElseThrow().status)
        assertEquals(RemovalRequestStatus.RESOLVING, original.status)
        assertEquals(RemovalItemState.EXCLUDED, items.findById(selected.id).orElseThrow().state)
        assertEquals(RemovalItemState.VERIFIED, items.findById(overlapping.id).orElseThrow().state)
        assertEquals(RemovalState.REVIEW, photos.findById(photo.id).orElseThrow().removalState)
        val audit = history.findByRequestIdAndCommandId(created.id, command.commandId).single()
        assertEquals("DRAFT", audit.fromState)
        assertEquals("RESOLVING", audit.toState)
        assertEquals(EvidenceCleanupResult.RETRY_REQUIRED, evidence.purgeResolvedEvidence(created.id))
        doNothing().`when`(storage).delete(key)
        assertEquals(EvidenceCleanupResult.DELETED, evidence.purgeResolvedEvidence(created.id))
        verify(storage, atLeastOnce()).delete(key)
        val settled = requests.findById(created.id).orElseThrow()
        assertEquals(RemovalRequestStatus.CANCELLED, settled.status)
        assertNull(settled.evidenceKey)
        assertNotNull(settled.resolvedAt)
        assertTrue(history.existsByRequestIdAndAction(created.id, "EVIDENCE_PURGED"))
        assertTrue(history.existsByRequestIdAndAction(created.id, "CANCELLED"))
        assertEquals(original, service.cancel(admin.id, created.id, command))
        assertEquals(RemovalState.REVIEW, photos.findById(photo.id).orElseThrow().removalState)
    }

    @Test fun `cross case command race rejects loser before PUT and preserves winner provenance`() {
        val winner = create()
        val loser = create()
        val command = UUID.randomUUID()
        val loserThread = AtomicReference<Thread>()
        val loserChecks = AtomicInteger()
        val loserChecked = CountDownLatch(1)
        val continueLoser = CountDownLatch(1)
        val winnerWriting = CountDownLatch(1)
        val settleWinner = CountDownLatch(1)
        doAnswer {
            // Derived repository methods are abstract on the spy target; use the real PK query.
            val result = commands.findById(command).orElse(null)
            if (Thread.currentThread() == loserThread.get() && loserChecks.incrementAndGet() == 2) {
                assertNull(result)
                loserChecked.countDown()
                assertTrue(continueLoser.await(10, TimeUnit.SECONDS))
            }
            result
        }.`when`(commands).findByCommandId(command)
        val startedKeys = java.util.concurrent.CopyOnWriteArrayList<String>()
        val provenance = java.util.concurrent.CopyOnWriteArrayList<Boolean>()
        `when`(storage.put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())).thenAnswer {
            val key = it.getArgument<String>(0)
            startedKeys.add(key)
            val receipt = commands.findByCommandId(command)!!
            provenance.add(receipt.requestId.toString() in key && receipt.intendedEvidenceKey == key && receipt.state == RemovalCommandState.PENDING)
            if (key.contains(winner.id.toString())) {
                winnerWriting.countDown()
                assertTrue(settleWinner.await(10, TimeUnit.SECONDS))
            }
            StoredObject(key, image().bytes.size.toLong(), "image/png")
        }
        val executor = Executors.newFixedThreadPool(2)
        try {
            val losing = executor.submit<Int> {
                loserThread.set(Thread.currentThread())
                try { evidence.replace(admin.id, loser.id, command, loser.version, image()); 200 }
                catch (error: ApiException) { error.status.value() }
            }
            if (!loserChecked.await(10, TimeUnit.SECONDS)) {
                fail("Losing worker did not reach claim lookup: ${losing.get(1, TimeUnit.SECONDS)}")
            }
            val winning = executor.submit<PhotoRemovalRequestDto> { evidence.replace(admin.id, winner.id, command, winner.version, image()) }
            assertTrue(winnerWriting.await(10, TimeUnit.SECONDS))
            continueLoser.countDown()
            assertEquals(409, losing.get(10, TimeUnit.SECONDS))
            val pending = commands.findByCommandId(command)!!
            assertEquals(winner.id, pending.requestId)
            assertEquals(admin.id, pending.actorIdSnapshot)
            assertEquals(RemovalCommandOperation.REPLACE_EVIDENCE, pending.operation)
            settleWinner.countDown()
            val original = winning.get(10, TimeUnit.SECONDS)
            val receipt = commands.findByCommandId(command)!!
            assertEquals(winner.id, receipt.requestId)
            assertEquals(RemovalCommandState.SUCCEEDED, receipt.state)
            assertEquals(original, mapper.treeToValue(receipt.result, PhotoRemovalRequestDto::class.java))
            assertEquals(original, evidence.replace(admin.id, winner.id, command, winner.version, image()))
            assertEquals(listOf("photo-removal/${winner.id}/evidence/$command"), startedKeys.toList())
            assertTrue(provenance.all { it })
            assertTrue(requests.findById(loser.id).orElseThrow().outstandingEvidenceKeys.isEmpty())
        } finally { continueLoser.countDown(); settleWinner.countDown(); executor.shutdownNow() }
    }

    @Test fun `identical replacement completed after preflight replays original result without second PUT`() {
        val case = create()
        val command = UUID.randomUUID()
        val delayedThread = AtomicReference<Thread>()
        val delayedChecks = AtomicInteger()
        val preflightCommitted = CountDownLatch(1)
        val continueValidation = CountDownLatch(1)
        doAnswer {
            val receipt = commands.findById(command).orElse(null)
            if (Thread.currentThread() == delayedThread.get() && delayedChecks.incrementAndGet() == 1) {
                assertNull(receipt)
                TransactionSynchronizationManager.registerSynchronization(object : TransactionSynchronization {
                    override fun afterCompletion(status: Int) {
                        assertEquals(TransactionSynchronization.STATUS_COMMITTED, status)
                        preflightCommitted.countDown()
                        assertTrue(continueValidation.await(10, TimeUnit.SECONDS))
                    }
                })
            }
            receipt
        }.`when`(commands).findByCommandId(command)
        val executor = Executors.newSingleThreadExecutor()
        try {
            val delayed = executor.submit<PhotoRemovalRequestDto> {
                delayedThread.set(Thread.currentThread())
                evidence.replace(admin.id, case.id, command, case.version, image())
            }
            assertTrue(preflightCommitted.await(10, TimeUnit.SECONDS))
            val original = evidence.replace(admin.id, case.id, command, case.version, image())
            continueValidation.countDown()
            assertEquals(original, delayed.get(10, TimeUnit.SECONDS))
            verify(storage, times(1)).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
            assertEquals(1, history.findByRequestIdAndCommandId(case.id, command).size)
            assertEquals(original, mapper.treeToValue(commands.findByCommandId(command)!!.result, PhotoRemovalRequestDto::class.java))
        } finally { continueValidation.countDown(); executor.shutdownNow() }
    }

    @Test fun `oversized synthetic PNG header is rejected before pixel decode or PUT`() {
        val case = create()
        val header = image().bytes.copyOf()
        ByteBuffer.wrap(header).putInt(16, 10_000).putInt(20, 10_000)
        val crc = CRC32().apply { update(header, 12, 17) }
        ByteBuffer.wrap(header).putInt(29, crc.value.toInt())
        assertTrue(header.size < 1024)
        assertTrue(ImagePixelGuard.exceedsPixelBudget(header))
        assertEquals(400, assertFailsWith<ApiException> {
            evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(header, "image/png", "oversized-header.png"))
        }.status.value())
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `valid synthetic WebP uploads after durable claim and streams exact private bytes`() {
        val case = create()
        val command = UUID.randomUUID()
        val image = webp()
        val key = "photo-removal/${case.id}/evidence/$command"
        `when`(storage.put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            val claim = commands.findByCommandId(command)!!
            assertEquals(case.id, claim.requestId)
            assertEquals(key, claim.intendedEvidenceKey)
            assertEquals(RemovalCommandState.PENDING, claim.state)
            assertTrue(key in requests.findById(case.id).orElseThrow().outstandingEvidenceKeys)
            assertContentEquals(image.bytes, it.getArgument<ByteArray>(1))
            assertEquals("image/webp", it.getArgument<String>(2))
            StoredObject(key, image.bytes.size.toLong(), "image/webp")
        }
        val replaced = evidence.replace(admin.id, case.id, command, case.version, image)
        assertEquals(1, replaced.evidenceGeneration)
        assertEquals(0, selfies.countByUserId(admin.id).toInt())
        `when`(storage.open(key)).thenAnswer { image.bytes.inputStream() }
        val streamed = get("/api/v1/admin/photo-removal-requests/${case.id}/evidence", admin)
        assertEquals(200, streamed.status)
        assertEquals("image/webp", streamed.response.headers().firstValue("Content-Type").orElse(""))
        assertTrue("no-store" in streamed.response.headers().firstValue("Cache-Control").orElse(""))
        assertContentEquals(image.bytes, streamed.response.body())
        verify(storage, times(1)).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `WebP validation rejects wrong format malformed absent metadata and truncated chunks without PUT`() {
        val case = create()
        val valid = webp().bytes
        val malformed = valid.copyOf().apply { this[20] = 0 }
        val oversizedChunk = valid.copyOf().apply { ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN).putInt(16, Int.MAX_VALUE) }
        val metadataAbsent = "RIFF".toByteArray() + byteArrayOf(4, 0, 0, 0) + "WEBP".toByteArray()
        for (bytes in listOf(image().bytes, malformed, metadataAbsent, valid.copyOf(24), oversizedChunk)) {
            assertEquals(400, assertFailsWith<ApiException> {
                evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(bytes, "image/webp", "invalid.webp"))
            }.status.value())
        }
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `WebP oversized declared dimensions stay bounded and fail before PUT`() {
        val case = create()
        val oversized = webp().bytes.copyOf()
        // VP8L header packs width-1 and height-1 in fourteen bits each; no pixel allocation.
        ByteBuffer.wrap(oversized).order(ByteOrder.LITTLE_ENDIAN).putInt(21, (9_999 or (9_999 shl 14)))
        assertTrue(oversized.size < 1024)
        assertEquals(400, assertFailsWith<ApiException> {
            evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(oversized, "image/webp", "oversized.webp"))
        }.status.value())
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `WebP canvas without a valid image payload never qualifies as evidence`() {
        val case = create()
        val canvas = ByteArray(10).apply { this[4] = 1; this[7] = 1 }
        for (bytes in listOf(webpContainer("VP8X" to canvas),
            webpContainer("VP8X" to canvas, "VP8L" to ByteArray(8)))) {
            assertEquals(400, assertFailsWith<ApiException> {
                evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(bytes, "image/webp", "no-image.webp"))
            }.status.value())
        }
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `synthetic animated WebP preserves accepted private original bytes`() {
        val case = create()
        val original = animatedWebp()
        val result = evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, original)
        assertEquals(1, result.evidenceGeneration)
        verify(storage).put(anyString(), org.mockito.AdditionalMatchers.aryEq(original.bytes) ?: byteArrayOf(), eq("image/webp") ?: "image/webp")
        assertEquals(0, selfies.countByUserId(admin.id).toInt())
    }

    @Test fun `WebP animated frames reject absent malformed truncated and overdeclared image chunks before PUT`() {
        val case = create()
        val canvas = ByteArray(10).apply { this[0] = 2; this[4] = 1; this[7] = 1 }
        val frame = ByteArray(16).apply { this[6] = 1; this[9] = 1; this[12] = 100 }
        val invalidSignature = frame + "VP8L".toByteArray() + byteArrayOf(8, 0, 0, 0) + ByteArray(8)
        val overdeclared = animatedWebp().bytes.copyOf().apply {
            ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN).putInt(72, Int.MAX_VALUE)
        }
        for (bytes in listOf(webpContainer("VP8X" to canvas, "ANIM" to ByteArray(6), "ANMF" to frame),
            webpContainer("VP8X" to canvas, "ANIM" to ByteArray(6), "ANMF" to frame.copyOf(15)),
            webpContainer("VP8X" to canvas, "ANIM" to ByteArray(6), "ANMF" to invalidSignature), overdeclared)) {
            assertEquals(400, assertFailsWith<ApiException> {
                evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(bytes, "image/webp", "invalid-frame.webp"))
            }.status.value())
        }
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `WebP oversized animated frame header is rejected without allocating pixels or PUT`() {
        val case = create()
        val bytes = animatedWebp().bytes.copyOf()
        // First ANMF frame body starts at52; declared width/height-1 occupy bytes6..11.
        for (offset in listOf(58, 61)) {
            bytes[offset] = (9_999 and 255).toByte()
            bytes[offset + 1] = (9_999 shr 8).toByte()
            bytes[offset + 2] = 0
        }
        assertTrue(bytes.size < 1024)
        assertEquals(400, assertFailsWith<ApiException> {
            evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(bytes, "image/webp", "oversized-frame.webp"))
        }.status.value())
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `valid WebP with trailing empty unknown chunks preserves accepted bytes`() {
        val case = create()
        val bytes = webpContainer("VP8L" to webp().bytes.copyOfRange(20, 37),
            "JUNK" to byteArrayOf(), "TEST" to byteArrayOf())
        val result = evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version,
            SearchImage(bytes, "image/webp", "unknown-chunks.webp"))
        assertEquals(1, result.evidenceGeneration)
        verify(storage).put(anyString(), org.mockito.AdditionalMatchers.aryEq(bytes) ?: byteArrayOf(), eq("image/webp") ?: "image/webp")
    }

    @Test fun `lossy WebP image metadata requires positive dimensions and preserves valid bytes`() {
        val case = create()
        // Locally encoded synthetic 2x2 RGB(20,40,60) using installed Pillow/libwebp.
        val bytes = java.util.Base64.getDecoder().decode("UklGRjAAAABXRUJQVlA4ICQAAABQAQCdASoCAAIAAUAmJQBOgCgAAP76id+R2EN2HLri5shvAAA=")
        val zeroWidth = bytes.copyOf().apply { this[26] = 0; this[27] = 0 }
        assertEquals(400, assertFailsWith<ApiException> {
            evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(zeroWidth, "image/webp", "zero-width.webp"))
        }.status.value())
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
        val accepted = evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, SearchImage(bytes, "image/webp", "synthetic-lossy.webp"))
        assertEquals(1, accepted.evidenceGeneration)
        verify(storage).put(anyString(), org.mockito.AdditionalMatchers.aryEq(bytes) ?: byteArrayOf(), eq("image/webp") ?: "image/webp")
    }

    @Test fun `evidence validates empty type bytes dimensions and five MB limit before storage`() {
        val case = create()
        for ((image, expected) in listOf(SearchImage(byteArrayOf(), "image/png", "empty.png") to 400,
            SearchImage(byteArrayOf(1), "text/plain", "bad.txt") to 400,
            SearchImage(byteArrayOf(1, 2, 3), "image/png", "bad.png") to 400,
            SearchImage(ByteArray(5 * 1024 * 1024 + 1), "image/jpeg", "large.jpg") to 413)) {
            assertEquals(expected, assertFailsWith<ApiException> { evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, image) }.status.value())
        }
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    @Test fun `list history and HTTP intake stay bounded and validate UUIDs and notes`() {
        val case = create()
        val page = queries.list(admin.id, event.id, null, null, 0, 500)
        assertEquals(100, page.limit)
        assertEquals(listOf(case.id), page.items.map { it.id })
        val historyPage = queries.history(admin.id, case.id, 0, 500)
        assertEquals(100, historyPage.limit)
        assertEquals("CREATED", historyPage.items.single().action)
        assertEquals(400, get("/api/v1/admin/photo-removal-requests/not-a-uuid", admin).status)
        val payload = mapper.writeValueAsString(intake().copy(adminNote = "x".repeat(2001)))
        assertEquals(400, post("/api/v1/admin/photo-removal-requests", payload, admin).status)
    }

    @Test fun `list reports phase counts without embedding case items`() {
        val case = create()
        items.saveAndFlush(PhotoRemovalItem(requestId = case.id, photoIdSnapshot = UUID.randomUUID(),
            state = RemovalItemState.CANDIDATE, matchSource = RemovalMatchSource.MANUAL))
        val summary = queries.list(admin.id, event.id, null, null, 0, 50).items.single()
        assertEquals(1, summary.phaseCounts.candidate)
        assertTrue(summary.items.isEmpty())
    }

    @Test fun `confirmed evidence purge is replayable while later resolution stays open`() {
        val case = create()
        val request = requests.findById(case.id).orElseThrow()
        request.status = RemovalRequestStatus.RESOLVING
        requests.saveAndFlush(request)
        assertEquals(EvidenceCleanupResult.DELETED, evidence.purgeResolvedEvidence(case.id))
        assertEquals(EvidenceCleanupResult.DELETED, evidence.purgeResolvedEvidence(case.id))
        assertNull(requests.findById(case.id).orElseThrow().resolvedAt)
        assertEquals(1, queries.history(admin.id, case.id, 0, 50).items.count { it.action == "EVIDENCE_PURGED" })
    }

    @Test fun `cancellation racing pending PUT keeps provenance until the writer settles`() {
        val case = create()
        val entered = CountDownLatch(1)
        val complete = CountDownLatch(1)
        val command = UUID.randomUUID()
        val key = "photo-removal/${case.id}/evidence/$command"
        `when`(storage.put(eq(key) ?: "", any(ByteArray::class.java) ?: byteArrayOf(), eq("image/png") ?: "")).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            entered.countDown()
            assertTrue(complete.await(10, TimeUnit.SECONDS))
            StoredObject(key, image().bytes.size.toLong(), "image/png")
        }
        val executor = Executors.newSingleThreadExecutor()
        try {
            val writer = executor.submit<Int> {
                try { evidence.replace(admin.id, case.id, command, case.version, image()); 200 }
                catch (error: ApiException) { error.status.value() }
            }
            assertTrue(entered.await(10, TimeUnit.SECONDS))
            val pending = queries.adminDetail(admin.id, case.id)
            assertEquals(409, assertFailsWith<ApiException> {
                evidence.replace(admin.id, case.id, UUID.randomUUID(), pending.version, image())
            }.status.value())
            service.cancel(admin.id, case.id, CancelRemovalCommand(UUID.randomUUID(), pending.version, "Withdraw pending upload"))
            assertEquals(EvidenceCleanupResult.RETRY_REQUIRED, evidence.purgeResolvedEvidence(case.id))
            assertNull(requests.findById(case.id).orElseThrow().resolvedAt)
            verify(storage, never()).delete(key)
            complete.countDown()
            assertEquals(409, writer.get(10, TimeUnit.SECONDS))
            val settled = requests.findById(case.id).orElseThrow()
            assertEquals(RemovalRequestStatus.CANCELLED, settled.status)
            assertNull(settled.evidenceKey)
            assertNotNull(settled.resolvedAt)
            assertTrue(settled.outstandingEvidenceKeys.isEmpty())
        } finally { complete.countDown(); executor.shutdownNow() }
    }

    @Test fun `replacement excludes old candidates while preserving verification and owner deadline`() {
        val case = create()
        val verifier = UUID.randomUUID()
        val verifiedAt = Instant.parse("2026-10-05T00:00:00Z")
        val request = requests.findById(case.id).orElseThrow()
        request.status = RemovalRequestStatus.OPEN
        request.verifierId = verifier
        request.verifiedAt = verifiedAt
        request.verificationNote = "Earlier explicit verification"
        val updated = requests.saveAndFlush(request)
        val candidate = items.saveAndFlush(PhotoRemovalItem(requestId = case.id, photoIdSnapshot = UUID.randomUUID(), matchSource = RemovalMatchSource.SELFIE))
        val confirmed = items.saveAndFlush(PhotoRemovalItem(requestId = case.id, photoIdSnapshot = UUID.randomUUID(),
            state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL))
        val deadline = Instant.parse("2026-10-08T00:00:00Z")
        val review = reviews.saveAndFlush(PhotoRemovalReview(requestId = case.id, ownerIdSnapshot = user(Role.PHOTOGRAPHER).id, deadline = deadline))
        val replaced = evidence.replace(admin.id, case.id, UUID.randomUUID(), updated.version, image())
        assertEquals(RemovalItemState.EXCLUDED, items.findById(candidate.id).orElseThrow().state)
        assertEquals(RemovalItemState.VERIFIED, items.findById(confirmed.id).orElseThrow().state)
        assertEquals(verifier, replaced.verifierId)
        assertEquals(verifiedAt, replaced.verifiedAt)
        assertEquals("Earlier explicit verification", replaced.verificationNote)
        assertEquals(deadline, reviews.findById(review.id).orElseThrow().deadline)
        assertEquals(1, replaced.evidenceGeneration)
    }

    @Test fun `replaced keys remain durable and retry sweep respects backoff`() {
        val case = create()
        val firstCommand = UUID.randomUUID()
        val first = evidence.replace(admin.id, case.id, firstCommand, case.version, image())
        val oldKey = "photo-removal/${case.id}/evidence/$firstCommand"
        evidence.replace(admin.id, case.id, UUID.randomUUID(), first.version, image())
        assertTrue(oldKey in requests.findById(case.id).orElseThrow().outstandingEvidenceKeys)
        doThrow(IllegalStateException("synthetic old-key delete failure")).`when`(storage).delete(oldKey)
        evidence.retryEvidenceCleanup()
        val failed = requests.findById(case.id).orElseThrow()
        assertTrue(oldKey in failed.outstandingEvidenceKeys)
        assertTrue(failed.evidenceCleanupNextAttemptAt!!.isAfter(Instant.now()))
        evidence.retryEvidenceCleanup()
        verify(storage, times(1)).delete(oldKey)
        failed.evidenceCleanupNextAttemptAt = Instant.now().minusSeconds(1)
        requests.saveAndFlush(failed)
        doNothing().`when`(storage).delete(oldKey)
        evidence.retryEvidenceCleanup()
        val cleaned = requests.findById(case.id).orElseThrow()
        assertTrue(cleaned.outstandingEvidenceKeys.isEmpty())
        assertNotNull(cleaned.evidenceKey)
        assertNull(cleaned.evidenceCleanupNextAttemptAt)
        assertEquals(RemovalEvidenceCleanupState.PRESENT, cleaned.evidenceCleanupState)
    }

    @Test fun `multipart enforces UUID version binding and size limits before storage`() {
        val case = create()
        val path = "/api/v1/admin/photo-removal-requests/${case.id}/evidence"
        assertEquals(400, upload(path, byteArrayOf(1), "not-a-uuid", "0").status)
        assertEquals(400, upload(path, image().bytes, UUID.randomUUID().toString(), "-1").status)
        assertEquals(413, upload(path, ByteArray(5 * 1024 * 1024 + 1), UUID.randomUUID().toString(), "0").status)
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
        val success = upload(path, image().bytes, UUID.randomUUID().toString(), "0")
        assertEquals(200, success.status)
        assertTrue("no-store" in success.response.headers().firstValue("Cache-Control").orElse(""))
    }

    @Test fun `outer transactions cannot hold a connection across evidence I O`() {
        val case = create()
        assertFailsWith<IllegalTransactionStateException> {
            tx.executeWithoutResult { evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, image()) }
        }
        assertEquals(case.version, requests.findById(case.id).orElseThrow().version)
        verify(storage, never()).put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())
    }

    private fun upload(path: String, bytes: ByteArray, command: String, version: String): Response {
        val boundary = "task4-${UUID.randomUUID()}"
        val output = ByteArrayOutputStream()
        for ((name, value) in listOf("commandId" to command, "expectedVersion" to version)) {
            output.write("--$boundary\r\nContent-Disposition: form-data; name=\"$name\"\r\n\r\n$value\r\n".toByteArray())
        }
        output.write("--$boundary\r\nContent-Disposition: form-data; name=\"selfie\"; filename=\"synthetic.png\"\r\nContent-Type: image/png\r\n\r\n".toByteArray())
        output.write(bytes)
        output.write("\r\n--$boundary--\r\n".toByteArray())
        return send(HttpRequest.newBuilder(URI("http://localhost:$port$path"))
            .header("Content-Type", "multipart/form-data; boundary=$boundary")
            .PUT(HttpRequest.BodyPublishers.ofByteArray(output.toByteArray())), admin)
    }

    private fun create() = service.create(admin.id, intake())
    private fun intake() = CreateRemovalRequestCommand(event.id, "support-${UUID.randomUUID()}", "Synthetic contact", null, "Privacy request", UUID.randomUUID())
    private fun user(role: Role) = users.saveAndFlush(User(email = "intake-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic ${role.name}", role = role))
    private fun image(): SearchImage {
        val bytes = ByteArrayOutputStream()
        ImageIO.write(BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), "png", bytes)
        return SearchImage(bytes.toByteArray(), "image/png", "synthetic.png")
    }
    // Encoded locally with installed Pillow/libwebp: synthetic solid RGB (20,40,60), 2x2 lossless.
    private fun webp() = SearchImage(java.util.Base64.getDecoder().decode(
        "UklGRh4AAABXRUJQVlA4TBEAAAAvAUAAAAdQlFKUp/+BiOh/AAA="), "image/webp", "synthetic.webp")
    private fun animatedWebp() = SearchImage(java.util.Base64.getDecoder().decode(
        "UklGRogAAABXRUJQVlA4WAoAAAACAAAAAQAAAQAAQU5JTQYAAAAAAAAAAABBTk1GKgAAAAAAAAAAAAEAAAEAAGQAAAJWUDhMEQAAAC8BQAAAB1CUUpSn/4GI6H8AAEFOTUYqAAAAAAAAAAAAAQAAAQAAZAAAAFZQOEwRAAAALwFAAAAHUJTylKL/gYjofwAA"),
        "image/webp", "synthetic-animated.webp")
    private fun webpContainer(vararg chunks: Pair<String, ByteArray>): ByteArray {
        val body = ByteArrayOutputStream().apply {
            write("WEBP".toByteArray())
            chunks.forEach { (name, bytes) ->
                write(name.toByteArray())
                write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(bytes.size).array())
                write(bytes)
                if (bytes.size % 2 == 1) write(0)
            }
        }.toByteArray()
        return "RIFF".toByteArray() + ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(body.size).array() + body
    }
    private data class Response(val response: HttpResponse<ByteArray>) { val status get() = response.statusCode() }
    private fun get(path: String, actor: User? = null): Response = send(HttpRequest.newBuilder(URI("http://localhost:$port$path")).GET(), actor)
    private fun post(path: String, body: String, actor: User): Response = send(HttpRequest.newBuilder(URI("http://localhost:$port$path"))
        .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)), actor)
    private fun send(builder: HttpRequest.Builder, actor: User?): Response {
        actor?.let { builder.header("Authorization", "Bearer ${tokens.createAccessToken(it)}") }
        return Response(client.send(builder.build(), HttpResponse.BodyHandlers.ofByteArray()))
    }
}
