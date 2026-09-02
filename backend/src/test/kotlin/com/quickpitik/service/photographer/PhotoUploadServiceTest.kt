package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
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
import java.sql.SQLException
import java.time.LocalDate
import java.time.ZoneId
import java.util.Optional
import java.util.UUID
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
    private lateinit var eventPhotographerRepository: EventPhotographerRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var eventPublisher: ApplicationEventPublisher
    private val storageProperties = StorageProperties()

    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()

    // "hello" has a well-known SHA-256, so asserting against it proves the
    // service hashes the raw bytes correctly (independent of MessageDigest).
    private val helloBytes = "hello".toByteArray(Charsets.UTF_8)
    private val helloSha256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    // Mockito.any() returns null at runtime; declared as a (platform) T so Kotlin
    // inserts no null-assertion — usable as a matcher for a non-null param.
    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> capture(c: ArgumentCaptor<T>): T = c.capture()

    @BeforeEach
    fun setUp() {
        storageService = Mockito.mock(StorageService::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventPhotographerRepository = Mockito.mock(EventPhotographerRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
    }

    private fun service(aiApiProperties: AiApiProperties) = PhotoUploadService(
        storageService,
        storageProperties,
        photoRepository,
        eventPhotographerRepository,
        eventRepository,
        photographerSettingsRepository,
        userRepository,
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
        Mockito.verify(photoRepository, Mockito.never()).saveAndFlush(anyArg<Photo>())
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
        Mockito.verify(photoRepository, Mockito.never()).saveAndFlush(anyArg<Photo>())
        Mockito.verify(storageService, Mockito.never())
            .put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
    }

    @Test
    fun `direct upload begin falls back to multipart when storage cannot presign PUTs`() {
        stubValidationsPass()
        Mockito.`when`(storageService.supportsDirectUpload).thenReturn(false)
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)

        val res = service(AiApiProperties(enabled = false))
            .beginDirectUpload(photographerId, eventId, helloSha256, "image/jpeg", 1_000)

        assertEquals("multipart", res.mode)
        Mockito.verify(storageService, Mockito.never()).presignedPutUrl(anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `direct upload begin issues a presigned PUT and commit persists the row`() {
        stubValidationsPass()
        Mockito.`when`(storageService.supportsDirectUpload).thenReturn(true)
        Mockito.`when`(storageService.presignedPutUrl(anyArg(), anyArg(), anyArg())).thenReturn("https://put")
        Mockito.`when`(storageService.exists(anyArg())).thenReturn(true)
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)
        val svc = service(AiApiProperties(enabled = false))

        val begin = svc.beginDirectUpload(photographerId, eventId, helloSha256, "image/jpeg", 1_000)
        assertEquals("direct", begin.mode)
        assertEquals("https://put", begin.uploadUrl)
        assertEquals("events/$eventId/photos/${begin.photoId}/original.jpg", begin.key)

        val dto = svc.commitDirectUpload(photographerId, eventId, begin.photoId!!, begin.key!!, helloSha256)

        val captor = ArgumentCaptor.forClass(Photo::class.java)
        Mockito.verify(photoRepository).saveAndFlush(capture(captor))
        assertEquals(helloSha256, captor.value.contentHash)
        assertEquals(begin.key, captor.value.s3Key)
        assertEquals("processing", dto.status)
        // The bytes never touched this server: no storage PUT in either step.
        Mockito.verify(storageService, Mockito.never()).put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
        Mockito.verify(eventPublisher).publishEvent(anyArg<PhotoUploadedForWatermark>())
    }

    @Test
    fun `direct upload commit refuses a key that was not issued for the photo`() {
        stubValidationsPass()
        val ex = org.junit.jupiter.api.assertThrows<ApiException> {
            service(AiApiProperties(enabled = false))
                .commitDirectUpload(photographerId, eventId, UUID.randomUUID(), "events/other/original.jpg", helloSha256)
        }
        assertEquals(ErrorCodes.UPLOAD_KEY_MISMATCH, ex.code)
        Mockito.verify(photoRepository, Mockito.never()).saveAndFlush(anyArg<Photo>())
    }

    @Test
    fun `first-time upload computes and stores the content hash`() {
        stubValidationsPass()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")

        val dto = service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())

        val captor = ArgumentCaptor.forClass(Photo::class.java)
        Mockito.verify(photoRepository).saveAndFlush(capture(captor))
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

    // The race-safe backstop: when a concurrent identical-bytes upload slips past
    // the pre-check and the (photographer_id, content_hash) unique index trips on
    // saveAndFlush, the violation is translated to a terminal duplicate conflict
    // rather than a 500. Postgres has aborted the transaction by then, so the
    // service can't re-query to tell same- from different-event — it emits the
    // same-event conflict and relies on the retry's pre-check for precision.
    @Test
    fun `race-lost insert on the dedup index surfaces a same-event duplicate conflict`() {
        stubValidationsPass()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)
        Mockito.`when`(photoRepository.saveAndFlush(anyArg<Photo>()))
            .thenThrow(dataIntegrityViolation("uq_photos_photographer_content_hash"))

        val ex = assertFailsWith<ConflictException> {
            service(AiApiProperties(enabled = false)).upload(photographerId, eventId, file())
        }
        assertEquals("PHOTO_DUPLICATE_SAME_EVENT", ex.code)
    }

    // A genuinely-unexpected constraint (e.g. a future FK) must surface as the
    // integrity fault it is — NOT be swallowed and mislabeled a duplicate.
    @Test
    fun `a non-dedup integrity violation is rethrown, not mislabeled a duplicate`() {
        stubValidationsPass()
        Mockito.`when`(photoRepository.findFirstByPhotographerIdAndContentHash(anyArg(), anyArg()))
            .thenReturn(null)
        Mockito.`when`(photoRepository.saveAndFlush(anyArg<Photo>()))
            .thenThrow(dataIntegrityViolation("fk_photos_some_other_constraint"))

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
