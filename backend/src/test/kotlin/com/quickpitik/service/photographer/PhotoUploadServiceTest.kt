package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotoUploadIntent
import com.quickpitik.entity.PhotoUploadIntentState
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoUploadIntentRepository
import com.quickpitik.repository.PhotoUploadQuota
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.PresignedUpload
import com.quickpitik.service.storage.StoredObject
import com.quickpitik.service.image.PhotoUploadValidator
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import jakarta.persistence.EntityManager
import org.hibernate.exception.ConstraintViolationException
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.web.multipart.MultipartFile
import java.math.BigDecimal
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.sql.SQLException
import java.security.MessageDigest
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.Optional
import java.util.HexFormat
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Duplicate detection on upload: a photo's identity is the SHA-256 of its
// original bytes, deduped per-photographer across all events. A same-event
// re-upload returns the existing photo with no side effects (idempotent); a
// different-event re-upload is rejected; a first-time upload stores the hash.
class PhotoUploadServiceTest {

    private lateinit var storageService: StorageService
    private lateinit var photoRepository: PhotoRepository
    private lateinit var photoUploadIntentRepository: PhotoUploadIntentRepository
    private lateinit var eventPhotographerRepository: EventPhotographerRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var eventPublisher: ApplicationEventPublisher
    private lateinit var entityManager: EntityManager
    private val storageProperties = StorageProperties(
        backend = StorageProperties.Backend.S3,
        endpoint = "https://test-account.r2.cloudflarestorage.com",
        directUpload = StorageProperties.DirectUpload(enabled = true, websiteEnabled = true),
    )

    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()

    private val helloBytes = ByteArrayOutputStream().use { output ->
        check(ImageIO.write(BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB), "jpg", output))
        output.toByteArray()
    }
    private val helloSha256 =
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(helloBytes))

    // Mockito.any() returns null at runtime; declared as a (platform) T so Kotlin
    // inserts no null-assertion — usable as a matcher for a non-null param.
    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> capture(c: ArgumentCaptor<T>): T = c.capture()

    @BeforeEach
    fun setUp() {
        storageService = Mockito.mock(StorageService::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        photoUploadIntentRepository = Mockito.mock(PhotoUploadIntentRepository::class.java)
        eventPhotographerRepository = Mockito.mock(EventPhotographerRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
        entityManager = Mockito.mock(EntityManager::class.java)
    }

    private fun service(aiApiProperties: AiApiProperties) = PhotoUploadService(
        storageService,
        storageProperties,
        photoRepository,
        photoUploadIntentRepository,
        eventPhotographerRepository,
        eventRepository,
        photographerSettingsRepository,
        PhotographerAccessGuard(userRepository, photographerSettingsRepository),
        PhotoUploadValidator(),
        aiApiProperties,
        eventPublisher,
        // Real template over a stubbed manager: execute { } runs the callback
        // inline (and rethrows), keeping these unit tests transaction-free. The
        // getTransaction stub matters — a raw mock returns a null status, which
        // trips the Kotlin lambda's non-null parameter check.
        TransactionTemplate(
            Mockito.mock(PlatformTransactionManager::class.java).also {
                Mockito.`when`(it.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
            },
        ),
        SimpleMeterRegistry(),
        entityManager,
        java.time.Clock.systemUTC(),
    )

    private fun event(id: UUID = eventId, name: String = "Cebu Marathon 2026"): Event =
        Event(
            id = id,
            slug = "e-$id",
            name = name,
            date = LocalDate.now(ZoneId.of("Asia/Manila")),
            location = "Cebu",
            status = EventStatus.ACTIVE,
            pricePerPhoto = BigDecimal("199.00"),
        )

    private fun photographer(): User =
        User(
            id = photographerId,
            email = "p@test.local",
            passwordHash = "x",
            name = "Photog",
            role = Role.PHOTOGRAPHER,
        )

    private fun approvedSettings(): PhotographerSettings =
        PhotographerSettings(
            userId = photographerId,
            watermarkS3Key = "watermarks/$photographerId.png",
            verificationStatus = VerificationStatus.APPROVED,
        )

    private fun existingPhoto(inEvent: UUID): Photo = photoWithHash(inEvent, helloSha256)

    private fun photoWithHash(inEvent: UUID, hash: String): Photo =
        Photo(
            eventId = inEvent,
            photographerId = photographerId,
            s3Key = "events/$inEvent/photos/x/original.jpg",
            thumbnailS3Key = "events/$inEvent/photos/x/watermark.jpg",
            pricePhp = BigDecimal("199.00"),
            contentHash = hash,
        )

    private fun file(bytes: ByteArray = helloBytes): MultipartFile {
        val f = Mockito.mock(MultipartFile::class.java)
        Mockito.`when`(f.contentType).thenReturn("image/jpeg")
        Mockito.`when`(f.isEmpty).thenReturn(false)
        Mockito.`when`(f.bytes).thenReturn(bytes)
        return f
    }

    // Stubs the validation gauntlet (event ACTIVE + in upload window, verified
    // un-suspended photographer) so upload() reaches the dedup check.
    private fun stubValidationsPass() {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event()))
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(photographer()))
        Mockito.`when`(photographerSettingsRepository.findById(photographerId))
            .thenReturn(Optional.of(approvedSettings()))
        // The thumbnail presign now runs BEFORE the persist block (it needs no
        // transaction), so every test that reaches the storage path needs it
        // stubbed — an unstubbed mock returns null and trips the Kotlin
        // non-null check at the persist lambda. Individual tests may re-stub.
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")
    }

    private fun stubIntentAdmission() {
        Mockito.`when`(storageService.supportsImmutableDirectUpload).thenReturn(true)
        Mockito.`when`(storageService.presignedUpload(anyArg(), anyArg(), anyArg(), Mockito.anyLong()))
            .thenReturn(PresignedUpload("https://test-account.r2.cloudflarestorage.com/put", mapOf("Content-Type" to "image/jpeg")))
        Mockito.`when`(photographerSettingsRepository.findByUserIdForUpdate(photographerId))
            .thenReturn(approvedSettings())
        Mockito.`when`(photoUploadIntentRepository.activeQuota(anyArg(), anyArg())).thenReturn(
            object : PhotoUploadQuota {
                override val slots = 0L
                override val bytes = 0L
            },
        )
        Mockito.`when`(photoUploadIntentRepository.saveAndFlush(anyArg())).thenAnswer { it.getArgument(0) }
    }

    @Test
    fun `direct begin reserves an owner scoped staging intent before presigning`() {
        stubValidationsPass()
        stubIntentAdmission()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg())).thenReturn(null)
        val requestId = UUID.randomUUID()

        val result = service(AiApiProperties(enabled = false)).beginDirectUpload(
            photographerId,
            eventId,
            helloSha256,
            "image/jpeg",
            1_000,
            requestId,
            "website",
        )

        val intent = ArgumentCaptor.forClass(PhotoUploadIntent::class.java)
        Mockito.verify(photoUploadIntentRepository).saveAndFlush(capture(intent))
        assertEquals(requestId, intent.value.requestId)
        assertEquals("staging/photo-uploads/$photographerId/$eventId/${intent.value.photoId}/original", intent.value.stagingKey)
        assertEquals(intent.value.stagingKey, result.key)
        assertEquals(mapOf("Content-Type" to "image/jpeg"), result.uploadHeaders)
    }

    @Test
    fun `direct begin replays a matching request id without reserving another intent`() {
        stubValidationsPass()
        val requestId = UUID.randomUUID()
        val intent = intent(helloBytes.size.toLong(), requestId = requestId)
        Mockito.`when`(photoUploadIntentRepository.findByPhotographerIdAndRequestId(photographerId, requestId))
            .thenReturn(intent)
        Mockito.`when`(storageService.presignedUpload(anyArg(), anyArg(), anyArg(), Mockito.anyLong()))
            .thenReturn(PresignedUpload("https://test-account.r2.cloudflarestorage.com/put", emptyMap()))

        val result = service(AiApiProperties(enabled = false)).beginDirectUpload(
            photographerId,
            eventId,
            helloSha256,
            "image/jpeg",
            helloBytes.size.toLong(),
            requestId,
            "website",
        )

        assertEquals(intent.photoId, result.photoId)
        assertEquals(intent.stagingKey, result.key)
        Mockito.verify(photoUploadIntentRepository, Mockito.never()).saveAndFlush(anyArg())
    }

    @Test
    fun `direct begin rejects a request id replay with different metadata`() {
        stubValidationsPass()
        val requestId = UUID.randomUUID()
        Mockito.`when`(photoUploadIntentRepository.findByPhotographerIdAndRequestId(photographerId, requestId))
            .thenReturn(intent(helloBytes.size.toLong(), requestId = requestId))

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).beginDirectUpload(
                photographerId,
                eventId,
                "0".repeat(64),
                "image/jpeg",
                helloBytes.size.toLong(),
                requestId,
                "website",
            )
        }

        assertEquals(ErrorCodes.UPLOAD_REQUEST_CONFLICT, error.code)
        Mockito.verify(storageService, Mockito.never()).presignedUpload(anyArg(), anyArg(), anyArg(), Mockito.anyLong())
    }

    @Test
    fun `request id race replays the winner even when it filled the pending quota`() {
        stubValidationsPass()
        stubIntentAdmission()
        val requestId = UUID.randomUUID()
        val winner = intent(helloBytes.size.toLong(), requestId = requestId)
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg())).thenReturn(null)
        Mockito.`when`(photoUploadIntentRepository.findByPhotographerIdAndRequestId(photographerId, requestId))
            .thenReturn(null, winner)
        Mockito.`when`(photoUploadIntentRepository.activeQuota(anyArg(), anyArg())).thenReturn(
            object : PhotoUploadQuota {
                override val slots = storageProperties.directUpload.maxPendingPerUser.toLong()
                override val bytes = winner.sizeBytes
            },
        )

        val result = service(AiApiProperties(enabled = false)).beginDirectUpload(
            photographerId,
            eventId,
            helloSha256,
            "image/jpeg",
            helloBytes.size.toLong(),
            requestId,
            "website",
        )

        assertEquals(winner.photoId, result.photoId)
        Mockito.verify(photoUploadIntentRepository, Mockito.never()).saveAndFlush(anyArg())
    }

    @Test
    fun `direct begin maps presign failures without exposing provider details`() {
        stubValidationsPass()
        val requestId = UUID.randomUUID()
        val intent = intent(helloBytes.size.toLong(), requestId = requestId)
        Mockito.`when`(photoUploadIntentRepository.findByPhotographerIdAndRequestId(photographerId, requestId))
            .thenReturn(intent)
        Mockito.`when`(storageService.presignedUpload(anyArg(), anyArg(), anyArg(), Mockito.anyLong()))
            .thenThrow(IllegalStateException("provider-secret"))

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).beginDirectUpload(
                photographerId,
                eventId,
                helloSha256,
                "image/jpeg",
                helloBytes.size.toLong(),
                requestId,
            )
        }

        assertEquals(ErrorCodes.UPLOAD_STORAGE_UNAVAILABLE, error.code)
        assertEquals(5, error.retryAfterSeconds)
        assertTrue("provider-secret" !in error.message.orEmpty())
    }

    @Test
    fun `unissued direct commit is hidden and never reads storage`() {
        stubValidationsPass()
        val id = UUID.randomUUID()
        Mockito.`when`(photoUploadIntentRepository.findOwned(id, photographerId, eventId)).thenReturn(null)
        Mockito.`when`(photoRepository.findById(id)).thenReturn(Optional.empty())

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).commitDirectUpload(
                photographerId,
                eventId,
                id,
                "staging/photo-uploads/$photographerId/$eventId/$id/original",
                helloSha256,
                1_000,
                "image/jpeg",
            )
        }

        assertEquals(ErrorCodes.UPLOAD_NOT_FOUND, error.code)
        Mockito.verify(storageService, Mockito.never()).metadata(anyArg())
        Mockito.verify(storageService, Mockito.never()).open(anyArg())
        Mockito.verify(storageService, Mockito.never()).delete(anyArg())
    }

    @Test
    fun `owned intent rejects client metadata that differs from its reservation`() {
        stubValidationsPass()
        val intent = intent(sizeBytes = 1_000)
        Mockito.`when`(photoUploadIntentRepository.findOwned(intent.photoId, photographerId, eventId)).thenReturn(intent)

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).commitDirectUpload(
                photographerId,
                eventId,
                intent.photoId,
                intent.stagingKey,
                "0".repeat(64),
                intent.sizeBytes,
                intent.contentType,
            )
        }

        assertEquals(ErrorCodes.UPLOAD_REQUEST_CONFLICT, error.code)
        Mockito.verify(storageService, Mockito.never()).metadata(anyArg())
    }

    private fun intent(
        sizeBytes: Long,
        requestId: UUID? = UUID.randomUUID(),
        putExpiresAt: OffsetDateTime = OffsetDateTime.now().plusMinutes(15),
    ): PhotoUploadIntent {
        val id = UUID.randomUUID()
        val now = OffsetDateTime.now()
        return PhotoUploadIntent(
            photoId = id,
            photographerId = photographerId,
            eventId = eventId,
            requestId = requestId,
            contentHash = helloSha256,
            contentType = "image/jpeg",
            sizeBytes = sizeBytes,
            stagingKey = "staging/photo-uploads/$photographerId/$eventId/$id/original",
            originalKey = "events/$eventId/photos/$id/original.jpg",
            putExpiresAt = putExpiresAt,
            expiresAt = now.plusHours(24),
            createdAt = now,
        )
    }

    @Test
    fun `same-event duplicate returns the existing photo and writes nothing`() {
        stubValidationsPass()
        val existing = existingPhoto(inEvent = eventId)
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(existing)
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")

        val dto = service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())

        assertEquals(existing.id, dto.id)
        assertEquals("https://thumb", dto.thumbnailUrl)
        // No new photo, no S3 write, no events fired — the re-upload is a no-op.
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
        Mockito.verify(storageService, Mockito.never())
            .put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
    }

    @Test
    fun `different-event duplicate is rejected with a clear conflict`() {
        stubValidationsPass()
        val otherEventId = UUID.randomUUID()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(existingPhoto(inEvent = otherEventId))
        Mockito.`when`(eventRepository.findById(otherEventId))
            .thenReturn(Optional.of(event(otherEventId, name = "Other Event")))

        val ex = assertFailsWith<ConflictException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }

        assertEquals("PHOTO_DUPLICATE_DIFFERENT_EVENT", ex.code)
        assertTrue(ex.message!!.contains("Other Event"))
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
        Mockito.verify(storageService, Mockito.never())
            .put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
    }

    @Test
    fun `direct upload begin falls back to multipart when storage cannot presign PUTs`() {
        stubValidationsPass()
        Mockito.`when`(storageService.supportsImmutableDirectUpload).thenReturn(false)
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)

        val res = service(AiApiProperties(enabled = false))
            .beginDirectUpload(photographerId, eventId, helloSha256, "image/jpeg", 1_000)

        assertEquals("multipart", res.mode)
        Mockito.verify(storageService, Mockito.never()).presignedUpload(anyArg(), anyArg(), anyArg(), Mockito.anyLong())
    }

    @Test
    fun `direct begin does not issue a slot for an image type the runtime cannot decode`() {
        stubValidationsPass()

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).beginDirectUpload(
                photographerId,
                eventId,
                helloSha256,
                "image/webp",
                helloBytes.size.toLong(),
            )
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
        Mockito.verify(photoUploadIntentRepository, Mockito.never()).saveAndFlush(anyArg())
    }

    @Test
    fun `direct upload begin issues a presigned PUT and commit persists the row`() {
        stubValidationsPass()
        stubIntentAdmission()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)
        var saved: PhotoUploadIntent? = null
        Mockito.`when`(photoUploadIntentRepository.saveAndFlush(anyArg())).thenAnswer {
            it.getArgument<PhotoUploadIntent>(0).also { intent -> saved = intent }
        }
        Mockito.`when`(photoUploadIntentRepository.findOwned(anyArg(), anyArg(), anyArg())).thenAnswer { saved }
        Mockito.`when`(photoUploadIntentRepository.acquireVerification(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        Mockito.`when`(photoUploadIntentRepository.complete(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        val svc = service(AiApiProperties(enabled = false))

        val begin = svc.beginDirectUpload(photographerId, eventId, helloSha256, "image/jpeg", helloBytes.size.toLong())
        assertEquals("direct", begin.mode)
        assertTrue(begin.key!!.startsWith("staging/photo-uploads/$photographerId/$eventId/"))
        val intent = checkNotNull(saved)
        Mockito.`when`(storageService.metadata(intent.originalKey)).thenReturn(null)
        Mockito.`when`(storageService.metadata(intent.stagingKey))
            .thenReturn(StoredObject(intent.stagingKey, helloBytes.size.toLong(), "image/jpeg", etag = "etag"))
        Mockito.`when`(storageService.copyIfAbsent(intent.stagingKey, intent.originalKey, "etag", "image/jpeg"))
            .thenReturn(StoredObject(intent.originalKey, helloBytes.size.toLong(), "image/jpeg"))
        Mockito.`when`(storageService.open(intent.originalKey)).thenReturn(ByteArrayInputStream(helloBytes))

        val dto = svc.commitDirectUpload(photographerId, eventId, begin.photoId!!, begin.key!!, helloSha256)

        val captor = ArgumentCaptor.forClass(Photo::class.java)
        Mockito.verify(entityManager).persist(capture(captor))
        assertEquals(helloSha256, captor.value.contentHash)
        assertEquals(intent.originalKey, captor.value.s3Key)
        assertEquals("processing", dto.status)
        // The bytes never touched this server: no storage PUT in either step.
        Mockito.verify(storageService, Mockito.never()).put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
        Mockito.verify(eventPublisher).publishEvent(anyArg<PhotoUploadedForWatermark>())
    }

    @Test
    fun `direct commit rejects bad stored size without deleting outside the janitor`() {
        stubValidationsPass()
        for (size in listOf(0L, 500L, 60L * 1024 * 1024)) {
            val intent = intent(helloBytes.size.toLong())
            Mockito.`when`(photoUploadIntentRepository.findOwned(intent.photoId, photographerId, eventId)).thenReturn(intent)
            Mockito.`when`(photoUploadIntentRepository.acquireVerification(anyArg(), anyArg(), anyArg(), anyArg()))
                .thenReturn(1)
            Mockito.`when`(photoUploadIntentRepository.reject(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
                .thenReturn(1)
            Mockito.`when`(storageService.metadata(intent.originalKey))
                .thenReturn(StoredObject(intent.originalKey, size, "image/jpeg"))
            val error = assertFailsWith<ApiException> {
                service(AiApiProperties(enabled = false)).commitDirectUpload(
                    photographerId,
                    eventId,
                    intent.photoId,
                    intent.stagingKey,
                    helloSha256,
                    sizeBytes = intent.sizeBytes,
                    contentType = intent.contentType,
                )
            }
            assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, error.code)
        }
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
        Mockito.verify(storageService, Mockito.never()).delete(anyArg())
    }

    @Test
    fun `direct commit rejects stored type that differs from the reservation`() {
        stubValidationsPass()
        val intent = intent(helloBytes.size.toLong())
        Mockito.`when`(photoUploadIntentRepository.findOwned(intent.photoId, photographerId, eventId)).thenReturn(intent)
        Mockito.`when`(photoUploadIntentRepository.acquireVerification(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        Mockito.`when`(photoUploadIntentRepository.reject(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        Mockito.`when`(storageService.metadata(intent.originalKey))
            .thenReturn(StoredObject(intent.originalKey, intent.sizeBytes, "image/png"))

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).commitDirectUpload(
                photographerId,
                eventId,
                intent.photoId,
                intent.stagingKey,
                intent.contentHash,
            )
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
        Mockito.verify(storageService, Mockito.never()).open(anyArg())
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
    }

    @Test
    fun `direct commit hashes the immutable original instead of trusting declared metadata`() {
        stubValidationsPass()
        val intent = intent(helloBytes.size.toLong())
        val changed = helloBytes.copyOf().apply { this[lastIndex] = (this[lastIndex].toInt() xor 1).toByte() }
        Mockito.`when`(photoUploadIntentRepository.findOwned(intent.photoId, photographerId, eventId)).thenReturn(intent)
        Mockito.`when`(photoUploadIntentRepository.acquireVerification(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        Mockito.`when`(photoUploadIntentRepository.reject(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        Mockito.`when`(storageService.metadata(intent.originalKey))
            .thenReturn(StoredObject(intent.originalKey, intent.sizeBytes, intent.contentType))
        Mockito.`when`(storageService.open(intent.originalKey)).thenReturn(ByteArrayInputStream(changed))

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).commitDirectUpload(
                photographerId,
                eventId,
                intent.photoId,
                intent.stagingKey,
                intent.contentHash,
            )
        }

        assertEquals(ErrorCodes.UPLOAD_CHECKSUM_MISMATCH, error.code)
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
    }

    @Test
    fun `missing object after the put deadline expires only the current intent lease`() {
        stubValidationsPass()
        val intent = intent(helloBytes.size.toLong(), putExpiresAt = OffsetDateTime.now().minusMinutes(1))
        Mockito.`when`(photoUploadIntentRepository.findOwned(intent.photoId, photographerId, eventId)).thenReturn(intent)
        Mockito.`when`(photoUploadIntentRepository.acquireVerification(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        Mockito.`when`(photoUploadIntentRepository.expireMissingAfterPut(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)

        val error = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).commitDirectUpload(
                photographerId,
                eventId,
                intent.photoId,
                intent.stagingKey,
                intent.contentHash,
            )
        }

        assertEquals(ErrorCodes.UPLOAD_OBJECT_MISSING, error.code)
        Mockito.verify(photoUploadIntentRepository).expireMissingAfterPut(anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `a verifier that loses its lease resolves the committed receipt`() {
        stubValidationsPass()
        val intent = intent(helloBytes.size.toLong())
        val winner = existingPhoto(eventId)
        Mockito.`when`(photoUploadIntentRepository.findOwned(intent.photoId, photographerId, eventId))
            .thenReturn(intent)
        Mockito.`when`(photoUploadIntentRepository.acquireVerification(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(1)
        Mockito.`when`(storageService.metadata(intent.originalKey))
            .thenReturn(StoredObject(intent.originalKey, helloBytes.size.toLong(), "image/jpeg"))
        Mockito.`when`(storageService.open(intent.originalKey)).thenReturn(ByteArrayInputStream(helloBytes))
        Mockito.`when`(photoUploadIntentRepository.complete(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenAnswer {
                intent.state = PhotoUploadIntentState.COMMITTED
                intent.resultPhotoId = winner.id
                0
            }
        Mockito.`when`(photoRepository.findById(winner.id)).thenReturn(Optional.of(winner))

        val result = service(AiApiProperties(enabled = false)).commitDirectUpload(
            photographerId,
            eventId,
            intent.photoId,
            intent.stagingKey,
            intent.contentHash,
        )

        assertEquals(winner.id, result.id)
    }

    @Test
    fun `retrying direct commit never deletes the referenced original`() {
        stubValidationsPass()
        val existing = existingPhoto(eventId)
        existing.s3Key = "events/$eventId/photos/${existing.id}/original.jpg"
        Mockito.`when`(storageService.exists(existing.s3Key)).thenReturn(true)
        Mockito.`when`(photoRepository.findById(existing.id)).thenReturn(Optional.of(existing))
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(photographerId, helloSha256))
            .thenReturn(existing)

        val dto = service(AiApiProperties(enabled = false)).commitDirectUpload(
            photographerId, eventId, existing.id, existing.s3Key, helloSha256,
        )

        assertEquals(existing.id, dto.id)
        Mockito.verify(storageService, Mockito.never()).delete(anyArg())
        Mockito.verify(eventRepository, Mockito.never()).incrementPhotoCount(anyArg())
        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
    }

    @Test
    fun `direct commit cannot overwrite another photographers photo id`() {
        stubValidationsPass()
        val existing = existingPhoto(eventId).also { it.photographerId = UUID.randomUUID() }
        existing.s3Key = "events/$eventId/photos/${existing.id}/original.jpg"
        Mockito.`when`(storageService.exists(existing.s3Key)).thenReturn(true)
        Mockito.`when`(photoRepository.findById(existing.id)).thenReturn(Optional.of(existing))

        val ex = assertFailsWith<ApiException> {
            service(AiApiProperties(enabled = false)).commitDirectUpload(
                photographerId, eventId, existing.id, existing.s3Key, helloSha256,
            )
        }

        assertEquals(ErrorCodes.UPLOAD_NOT_FOUND, ex.code)
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
        Mockito.verify(storageService, Mockito.never()).delete(anyArg())
    }

    @Test
    fun `direct upload commit refuses a key that was not issued for the photo`() {
        stubValidationsPass()
        val ex = org.junit.jupiter.api.assertThrows<ApiException> {
            service(AiApiProperties(enabled = false))
                .commitDirectUpload(photographerId, eventId, UUID.randomUUID(), "events/other/original.jpg", helloSha256)
        }
        assertEquals(ErrorCodes.UPLOAD_NOT_FOUND, ex.code)
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
    }

    @Test
    fun `first-time upload computes and stores the content hash`() {
        stubValidationsPass()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")

        val dto = service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())

        val captor = ArgumentCaptor.forClass(Photo::class.java)
        Mockito.verify(entityManager).persist(capture(captor))
        assertEquals(helloSha256, captor.value.contentHash)
        // Async watermark (2026-08-28): the row is inserted PROCESSING with no
        // watermark keys — PhotoWatermarkService flips it LIVE off-request.
        assertEquals(PhotoStatus.PROCESSING, captor.value.status)
        assertEquals(null, captor.value.thumbnailS3Key)
        assertEquals("processing", dto.status)
        // Exactly ONE storage write in-request: the original. The watermark PUT
        // moved to the async pipeline.
        Mockito.verify(storageService, Mockito.times(1))
            .put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
        // The watermark job is queued for AFTER_COMMIT dispatch.
        Mockito.verify(eventPublisher).publishEvent(anyArg<PhotoUploadedForWatermark>())
    }

    @Test
    fun `checkExisting reports new, same-event, and different-event per hash`() {
        val sameEventHash = "1111111111111111111111111111111111111111111111111111111111111111"
        val otherEventHash = "2222222222222222222222222222222222222222222222222222222222222222"
        val newHash = "3333333333333333333333333333333333333333333333333333333333333333"
        val otherEventId = UUID.randomUUID()

        Mockito.`when`(photoRepository.findByPhotographerIdAndContentHashIn(anyArg(), anyArg()))
            .thenReturn(
                listOf(
                    photoWithHash(eventId, sameEventHash),
                    photoWithHash(otherEventId, otherEventHash),
                ),
            )
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>()))
            .thenReturn(listOf(event(otherEventId, name = "Cebu Night Run")))

        val resp = service(AiApiProperties(enabled = false))
            .checkExisting(photographerId, eventId, listOf(sameEventHash, otherEventHash, newHash))

        // Every requested hash gets exactly one result, classified by boundary.
        assertEquals(3, resp.results.size)
        val byHash = resp.results.associateBy { it.hash }
        assertEquals("same_event", byHash[sameEventHash]?.status)
        assertEquals("different_event", byHash[otherEventHash]?.status)
        assertEquals("Cebu Night Run", byHash[otherEventHash]?.eventName)
        assertEquals("new", byHash[newHash]?.status)
    }

    // ── Photographer-owned events (V46) ──────────────────────────────────
    // The owner uploads before/during and up to one month after the event date
    // (not the 4-day race-day window), but a pending event takes nothing and
    // nobody else may upload into their event. A NONE watermark policy is the
    // one case that needs no logo at all.

    private fun ownedEvent(
        date: LocalDate,
        status: EventStatus = EventStatus.ACTIVE,
        policy: WatermarkPolicy = WatermarkPolicy.PLATFORM,
        owner: UUID = photographerId,
    ): Event = Event(
        id = eventId,
        slug = "e-$eventId",
        name = "Own Run",
        date = date,
        location = "Cebu",
        status = status,
        pricePerPhoto = BigDecimal.ZERO,
        createdBy = owner,
        watermarkPolicy = policy,
    )

    private fun stubOwnedUpload(event: Event, settings: PhotographerSettings = approvedSettings()) {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event))
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(photographer()))
        Mockito.`when`(photographerSettingsRepository.findById(photographerId)).thenReturn(Optional.of(settings))
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg())).thenReturn(null)
    }

    @Test
    fun `the owner may upload to their live event outside the race-day window`() {
        stubOwnedUpload(ownedEvent(date = LocalDate.now(ZoneId.of("Asia/Manila")).minusDays(20)))

        val dto = service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())

        assertEquals("processing", dto.status)
        Mockito.verify(entityManager).persist(anyArg<Photo>())
    }

    @Test
    fun `another photographer cannot upload into an owned event`() {
        stubOwnedUpload(ownedEvent(date = LocalDate.now(ZoneId.of("Asia/Manila")), owner = UUID.randomUUID()))

        val ex = org.junit.jupiter.api.assertThrows<ApiException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }
        assertEquals(ErrorCodes.EVENT_NOT_UPLOADABLE, ex.code)
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
    }

    @Test
    fun `an owned event still awaiting review takes no uploads`() {
        stubOwnedUpload(ownedEvent(date = LocalDate.now(ZoneId.of("Asia/Manila")), status = EventStatus.DRAFT))

        val ex = org.junit.jupiter.api.assertThrows<ApiException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }
        assertEquals(ErrorCodes.EVENT_NOT_UPLOADABLE, ex.code)
    }

    @Test
    fun `a no-watermark event needs no logo`() {
        stubOwnedUpload(
            ownedEvent(date = LocalDate.now(ZoneId.of("Asia/Manila")), policy = WatermarkPolicy.NONE),
            settings = PhotographerSettings(userId = photographerId, watermarkS3Key = null, verificationStatus = VerificationStatus.APPROVED),
        )

        val dto = service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())

        assertEquals("processing", dto.status)
    }

    @Test
    fun `a platform-marked event without a logo is still refused`() {
        stubOwnedUpload(
            ownedEvent(date = LocalDate.now(ZoneId.of("Asia/Manila"))),
            settings = PhotographerSettings(userId = photographerId, watermarkS3Key = null, verificationStatus = VerificationStatus.APPROVED),
        )

        val ex = org.junit.jupiter.api.assertThrows<ApiException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }
        assertEquals(ErrorCodes.WATERMARK_MISSING, ex.code)
    }

    @Test
    fun `the owner may upload before the event date`() {
        stubOwnedUpload(ownedEvent(date = LocalDate.now(ZoneId.of("Asia/Manila")).plusDays(10)))

        val dto = service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())

        assertEquals("processing", dto.status)
        Mockito.verify(entityManager).persist(anyArg<Photo>())
    }

    @Test
    fun `the owner is closed out more than one month after the event date`() {
        stubOwnedUpload(ownedEvent(date = LocalDate.now(ZoneId.of("Asia/Manila")).minusMonths(1).minusDays(5)))

        val ex = org.junit.jupiter.api.assertThrows<ApiException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }
        assertEquals(ErrorCodes.EVENT_NOT_UPLOADABLE, ex.code)
        assertTrue(ex.message!!.contains("1-month"))
        Mockito.verify(entityManager, Mockito.never()).persist(anyArg<Photo>())
    }

    // Refactoring the owned window into an if/else must not loosen the admin
    // race-day window (createdBy == null): a past admin event is still closed.
    @Test
    fun `an admin event outside the race-day window is still closed`() {
        val adminEvent = event().apply { date = LocalDate.now(ZoneId.of("Asia/Manila")).minusDays(10) }
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(adminEvent))
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(photographer()))
        Mockito.`when`(photographerSettingsRepository.findById(photographerId)).thenReturn(Optional.of(approvedSettings()))
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg())).thenReturn(null)

        val ex = org.junit.jupiter.api.assertThrows<ApiException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }
        assertEquals(ErrorCodes.EVENT_NOT_UPLOADABLE, ex.code)
        assertTrue(ex.message!!.contains("has closed"))
    }

    // The race-safe backstop: when a concurrent identical-bytes upload slips past
    // the pre-check and the (photographer_id, content_hash) unique index trips on
    // saveAndFlush, the violation is translated to a terminal duplicate conflict
    // rather than a 500. Postgres has aborted the transaction by then, so the
    // service can't re-query to tell same- from different-event — it emits the
    // same-event conflict and relies on the retry's pre-check for precision.
    @Test
    fun `race-lost insert returns the committed photo after rollback`() {
        stubValidationsPass()
        val existing = existingPhoto(eventId)
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null, existing)
        Mockito.doThrow(dataIntegrityViolation("uq_photos_photographer_content_hash"))
            .`when`(entityManager).flush()

        val dto = service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        assertEquals(existing.id, dto.id)
        Mockito.verify(eventRepository, Mockito.never()).incrementPhotoCount(anyArg())
    }

    // A genuinely-unexpected constraint (e.g. a future FK) must surface as the
    // integrity fault it is — NOT be swallowed and mislabeled a duplicate.
    @Test
    fun `a non-dedup integrity violation is rethrown, not mislabeled a duplicate`() {
        stubValidationsPass()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)
        Mockito.doThrow(dataIntegrityViolation("fk_photos_some_other_constraint"))
            .`when`(entityManager).flush()

        assertFailsWith<DataIntegrityViolationException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }
    }

    // The truncated-image case (ImageIO IOException on decode) moved with the
    // watermark work into the async pipeline — see PhotoWatermarkServiceTest's
    // semantic-failure test. In-request, undecodable-but-header-valid bytes now
    // upload 200 and settle via the processing_attempts budget.

    // A Spring DataIntegrityViolationException wrapping a Hibernate
    // ConstraintViolationException that names the given constraint — the shape a
    // real Postgres unique/FK violation arrives in.
    private fun dataIntegrityViolation(constraintName: String): DataIntegrityViolationException =
        DataIntegrityViolationException(
            "constraint violation",
            ConstraintViolationException("constraint violation", SQLException("violation"), constraintName),
        )
}
