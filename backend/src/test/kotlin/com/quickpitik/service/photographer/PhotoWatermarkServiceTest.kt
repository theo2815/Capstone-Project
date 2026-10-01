package com.quickpitik.service.photographer

import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoPublishedEvent
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import javax.imageio.IIOException
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// The async watermark pipeline: PROCESSING photos get their watermark
// derivative generated off-request and flip LIVE via a targeted conditional
// UPDATE. Attempt semantics mirror indexing's 2026-08-27 hardening — only
// semantic failures (undecodable bytes) burn processing_attempts; transport
// failures leave the budget intact for the reconcile sweep.
class PhotoWatermarkServiceTest {

    private lateinit var storageService: StorageService
    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var settingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var watermarkService: WatermarkService
    private lateinit var eventPublisher: ApplicationEventPublisher
    private lateinit var metadataService: PhotoMetadataService
    private lateinit var transactionManager: PlatformTransactionManager
    private var lastOriginal: ByteArray? = null

    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private val photoId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()
    private var lastCredit: WatermarkCredit? = null
    private var lastLogo: ByteArray? = null
    private var lastPlatformMark: Boolean? = null
    private var lastPlacement: WatermarkPlacement? = null

    @BeforeEach
    fun setUp() {
        storageService = Mockito.mock(StorageService::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        watermarkService = Mockito.mock(WatermarkService::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
        metadataService = Mockito.mock(PhotoMetadataService::class.java)
        transactionManager = Mockito.mock(PlatformTransactionManager::class.java)
        Mockito.`when`(transactionManager.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
    }

    private fun service(maxAttempts: Int = 5, metadata: PhotoMetadataService = metadataService) = PhotoWatermarkService(
        storageService,
        StorageProperties(),
        photoRepository,
        eventRepository,
        settingsRepository,
        userRepository,
        watermarkService,
        // Real cache over the mocked storage — first get() always calls through.
        WatermarkLogoCache(storageService, SimpleMeterRegistry()),
        eventPublisher,
        TransactionTemplate(transactionManager),
        SimpleMeterRegistry(),
        maxAttempts,
        metadata,
    )

    private fun photo(
        status: PhotoStatus = PhotoStatus.PROCESSING,
        attempts: Int = 0,
    ): Photo = Photo(
        id = photoId,
        eventId = eventId,
        photographerId = photographerId,
        s3Key = "events/$eventId/photos/$photoId/original.jpg",
        pricePhp = BigDecimal("199.00"),
        status = status,
    ).also { it.processingAttempts = attempts }

    private fun settings(brandName: String? = null, logoKey: String? = "watermarks/logo.png") = PhotographerSettings(
        userId = photographerId,
        brandName = brandName,
        handle = "anareyes",
        watermarkS3Key = logoKey,
        verificationStatus = VerificationStatus.APPROVED,
    )

    // The event decides which marks the preview carries (V46).
    private fun stubEvent(policy: WatermarkPolicy = WatermarkPolicy.PLATFORM) {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(
            Optional.of(
                Event(
                    id = eventId,
                    slug = "e-$eventId",
                    name = "Fun Run",
                    date = LocalDate.of(2026, 9, 1),
                    location = "Cebu City",
                    status = EventStatus.ACTIVE,
                    watermarkPolicy = policy,
                ),
            ),
        )
    }

    private fun user() = User(
        id = photographerId,
        email = "ana@test.local",
        passwordHash = "\$2a\$12\$stub",
        name = "Ana Reyes",
        role = Role.PHOTOGRAPHER,
    )

    private fun stubHappyPath(brandName: String? = null, logoKey: String? = "watermarks/logo.png") {
        Mockito.`when`(photoRepository.findById(photoId)).thenReturn(Optional.of(photo()))
        stubEvent()
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(Optional.of(settings(brandName, logoKey)))
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(user()))
        Mockito.`when`(storageService.getBytes("events/$eventId/photos/$photoId/original.jpg"))
            .thenReturn(byteArrayOf(1))
        Mockito.`when`(storageService.getBytes("watermarks/logo.png")).thenReturn(byteArrayOf(2))
        Mockito.`when`(watermarkService.processThumbnail(anyArg(), anyArg(), anyArg(), Mockito.anyBoolean(), anyArg())).thenAnswer { inv ->
            // Kotlin's non-null parameter rejects ArgumentCaptor/eq (both return
            // null), so record the arguments here instead.
            lastOriginal = inv.getArgument(0)
            lastLogo = inv.getArgument(1)
            lastCredit = inv.getArgument(2)
            lastPlatformMark = inv.getArgument(3)
            lastPlacement = inv.getArgument(4)
            MarkedPreview(jpeg = byteArrayOf(3), phash = 42L, phashClean = 43L, phashCentre = 44L)
        }
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")
    }

    // ── Per-event watermark policy (V46) ──────────────────────────────────

    @Test
    fun `an in-flight worker removes a derivative written after permanent deletion`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.findById(photoId)).thenReturn(Optional.of(photo()), Optional.empty())

        service().process(photoId)

        Mockito.verify(storageService).delete("events/$eventId/photos/$photoId/watermark.jpg")
        Mockito.verify(photoRepository, Mockito.never()).publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())
        Mockito.verifyNoInteractions(eventPublisher)
    }

    @Test
    fun `a platform event composites both marks`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)

        service().process(photoId)

        assertEquals(true, lastPlatformMark)
        assertTrue(lastLogo!!.contentEquals(byteArrayOf(2)))
    }

    @Test
    fun `a free event with policy OWN keeps the logo and drops the platform mark`() {
        stubHappyPath()
        stubEvent(WatermarkPolicy.OWN)
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)

        service().process(photoId)

        assertEquals(false, lastPlatformMark)
        assertTrue(lastLogo!!.contentEquals(byteArrayOf(2)))
        Mockito.verify(photoRepository).publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())
    }

    @Test
    fun `a free event with policy NONE publishes an unmarked preview even without a logo pointer`() {
        stubHappyPath(logoKey = null)
        stubEvent(WatermarkPolicy.NONE)
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)

        service().process(photoId)

        assertEquals(false, lastPlatformMark)
        assertNull(lastLogo)
        Mockito.verify(storageService).put("events/$eventId/photos/$photoId/watermark.jpg", byteArrayOf(3), "image/jpeg")
        Mockito.verify(photoRepository).publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())
    }

    @Test
    fun `a platform event still parks the photo when the logo pointer is missing`() {
        stubHappyPath(logoKey = null)

        service().process(photoId)

        assertFalse(lastPlatformMark != null, "processThumbnail must not run without the logo on a PLATFORM event")
        Mockito.verify(photoRepository, Mockito.never()).publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())
        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
    }

    @Test
    fun `success stores the watermark, flips LIVE with its phash, and publishes photo-published`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)

        service().process(photoId)

        val watermarkKey = "events/$eventId/photos/$photoId/watermark.jpg"
        Mockito.verify(storageService).put(watermarkKey, byteArrayOf(3), "image/jpeg")
        Mockito.verify(photoRepository).publishWatermarked(photoId, watermarkKey, 42L, 43L, 44L)
        Mockito.verify(eventPublisher).publishEvent(anyArg<PhotoPublishedEvent>())
        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
    }

    @Test
    fun `metadata uses the same original after publication commits and storage failure stays isolated`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)
        var committed = false
        var published = false
        var extracted = false
        Mockito.doAnswer { published = true; null }.`when`(eventPublisher).publishEvent(anyArg<PhotoPublishedEvent>())
        Mockito.doAnswer { committed = true; null }.`when`(transactionManager).commit(anyArg())
        Mockito.doAnswer { invocation ->
            assertTrue(committed)
            assertTrue(published)
            assertTrue(lastOriginal === invocation.getArgument<ByteArray>(1))
            extracted = true
            throw IllegalStateException("metadata database unavailable")
        }.`when`(metadataService).dispatchExtraction(anyArg(), anyArg())

        service().process(photoId)

        assertTrue(extracted)
        Mockito.verify(storageService, Mockito.times(1)).getBytes("events/$eventId/photos/$photoId/original.jpg")
        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
        Mockito.verify(transactionManager, Mockito.never()).rollback(anyArg())
    }

    @Test
    fun `blocked metadata releases both watermark workers after publication`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)
        val metadata = Mockito.mock(com.quickpitik.repository.PhotoMetadataRepository::class.java)
        val extractor = Mockito.mock(com.quickpitik.service.image.PhotoMetadataExtractor::class.java)
        Mockito.`when`(metadata.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        val parsing = java.util.concurrent.CountDownLatch(1)
        val release = java.util.concurrent.CountDownLatch(1)
        val returned = java.util.concurrent.CountDownLatch(2)
        Mockito.`when`(extractor.extract(anyArg())).thenAnswer {
            parsing.countDown()
            assertTrue(release.await(10, java.util.concurrent.TimeUnit.SECONDS))
            com.quickpitik.dto.photos.PhotoMetadataFields()
        }
        val workers = com.quickpitik.config.AsyncConfig().watermarkProcessingExecutor()
        val metadataWorkers = com.quickpitik.config.AsyncConfig().metadataProcessingExecutor()
        val extraction = PhotoMetadataService(metadata, photoRepository,
            Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java), extractor, storageService,
            workers, com.quickpitik.config.PhotoMetadataProperties(), TransactionTemplate(transactionManager), java.time.Clock.systemUTC(), metadataWorkers)
        try {
            repeat(2) { workers.execute { service(metadata = extraction).process(photoId); returned.countDown() } }
            assertTrue(parsing.await(5, java.util.concurrent.TimeUnit.SECONDS))
            assertTrue(returned.await(1, java.util.concurrent.TimeUnit.SECONDS), "blocked metadata occupied watermark workers")
        } finally {
            release.countDown()
            workers.shutdown()
            metadataWorkers.shutdown()
        }
    }

    @Test
    fun `parser exception after committed publication leaves watermark successful`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)
        val metadata = Mockito.mock(com.quickpitik.repository.PhotoMetadataRepository::class.java)
        val extractor = Mockito.mock(com.quickpitik.service.image.PhotoMetadataExtractor::class.java)
        Mockito.`when`(metadata.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        var committed = false
        var parsed = false
        Mockito.doAnswer { committed = true; null }.`when`(transactionManager).commit(anyArg())
        Mockito.`when`(extractor.extract(anyArg())).thenAnswer {
            assertTrue(committed)
            parsed = true
            throw IllegalArgumentException("broken metadata")
        }
        val extraction = PhotoMetadataService(
            metadata,
            Mockito.mock(com.quickpitik.repository.PhotoRepository::class.java),
            Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java),
            extractor,
            Mockito.mock(com.quickpitik.service.storage.StorageService::class.java),
            Mockito.mock(org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor::class.java),
            com.quickpitik.config.PhotoMetadataProperties(),
            TransactionTemplate(transactionManager),
            java.time.Clock.systemUTC(),
            Mockito.mock(org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor::class.java).also { pool ->
                Mockito.doAnswer { it.getArgument<Runnable>(0).run(); null }.`when`(pool).execute(Mockito.any(Runnable::class.java))
            },
        )

        service(metadata = extraction).process(photoId)

        assertTrue(parsed)
        Mockito.verify(metadata).settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg(), Mockito.isNull<Long>())
        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
        Mockito.verify(transactionManager, Mockito.never()).rollback(anyArg())
    }

    @Test
    fun `failed publication commit never starts metadata extraction`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)
        Mockito.doThrow(IllegalStateException("commit failed")).`when`(transactionManager).commit(anyArg())

        kotlin.test.assertFailsWith<IllegalStateException> { service().process(photoId) }

        Mockito.verifyNoInteractions(metadataService)
    }

    // The credit baked into the preview: brand name when the photographer set
    // one, otherwise the account name; handle straight from settings.
    @Test
    fun `credit prefers the brand name and falls back to the account name`() {
        stubHappyPath(brandName = "Reyes Race Photos")
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)
        service().process(photoId)
        assertEquals(WatermarkCredit("Reyes Race Photos", "anareyes", photoId), lastCredit)

        stubHappyPath(brandName = "  ")
        service().process(photoId)
        assertEquals("Ana Reyes", lastCredit?.name)
    }

    // V51: the photographer's own logo placement (position per orientation,
    // opacity, size) is read straight off the settings row and threaded into
    // the compositor. Distinct values in every field so a mis-wired mapping is
    // caught. Applies under whatever policy draws the logo (here PLATFORM).
    @Test
    fun `placement is read from settings and passed to the compositor`() {
        stubHappyPath()
        val custom = settings().apply {
            watermarkOpacity = 0.5
            watermarkScale = 0.25
            watermarkPosLandscapeX = 0.125
            watermarkPosLandscapeY = 0.375
            watermarkPosPortraitX = 0.625
            watermarkPosPortraitY = 0.875
        }
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(Optional.of(custom))
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(1)

        service().process(photoId)

        assertEquals(
            WatermarkPlacement(
                opacity = 0.5f,
                scale = 0.25,
                landscapeX = 0.125,
                landscapeY = 0.375,
                portraitX = 0.625,
                portraitY = 0.875,
            ),
            lastPlacement,
        )
    }

    @Test
    fun `a lost flip race publishes no websocket frame`() {
        stubHappyPath()
        // Another worker (sweep vs hot path) flipped it first — 0 rows updated.
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())).thenReturn(0)

        service().process(photoId)

        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
        Mockito.verifyNoInteractions(metadataService)
    }

    @Test
    fun `a photo that is no longer PROCESSING is left alone`() {
        Mockito.`when`(photoRepository.findById(photoId))
            .thenReturn(Optional.of(photo(status = PhotoStatus.LIVE)))

        service().process(photoId)

        Mockito.verifyNoInteractions(storageService, watermarkService, eventPublisher)
    }

    @Test
    fun `an exhausted attempt budget stops the re-drive`() {
        Mockito.`when`(photoRepository.findById(photoId))
            .thenReturn(Optional.of(photo(attempts = 5)))

        service(maxAttempts = 5).process(photoId)

        Mockito.verifyNoInteractions(storageService, watermarkService, eventPublisher)
    }

    // The truncated-PTP-pull case that used to 422 in-request: the bytes in
    // storage never change, so retrying can never succeed — the attempt budget
    // must burn so the sweep eventually stops re-driving it.
    @Test
    fun `undecodable bytes burn an attempt and never flip the photo`() {
        stubHappyPath()
        // doAnswer, not thenThrow: Kotlin declares no checked exceptions, so
        // Mockito rejects thenThrow(IIOException) for this method.
        Mockito.doAnswer { throw IIOException("Image is truncated") }
            .`when`(watermarkService).processThumbnail(anyArg(), anyArg(), anyArg(), Mockito.anyBoolean(), anyArg())

        service().process(photoId)

        Mockito.verify(photoRepository).incrementProcessingAttempts(photoId)
        Mockito.verify(storageService, Mockito.never())
            .put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
        Mockito.verify(photoRepository, Mockito.never()).publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())
        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
        Mockito.verifyNoInteractions(metadataService)
    }

    @Test
    fun `a storage transport failure keeps the attempt budget intact`() {
        Mockito.`when`(photoRepository.findById(photoId)).thenReturn(Optional.of(photo()))
        stubEvent()
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(Optional.of(settings()))
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(user()))
        Mockito.`when`(storageService.getBytes(anyArg()))
            .thenThrow(RuntimeException("storage unreachable"))

        service().process(photoId)

        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
        Mockito.verify(photoRepository, Mockito.never()).publishWatermarked(anyArg(), anyArg(), Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong())
        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
        Mockito.verifyNoInteractions(metadataService)
    }

    // photographer_id is FK-backed, so a missing user row is an inconsistency,
    // not a property of the bytes. Treat it like the missing-logo case: leave
    // the budget alone and let the sweep re-drive once it's fixed.
    @Test
    fun `a missing user row is a transport failure`() {
        Mockito.`when`(photoRepository.findById(photoId)).thenReturn(Optional.of(photo()))
        stubEvent()
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(Optional.of(settings()))
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.empty())

        service().process(photoId)

        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
        Mockito.verifyNoInteractions(storageService, watermarkService, eventPublisher)
    }
}
