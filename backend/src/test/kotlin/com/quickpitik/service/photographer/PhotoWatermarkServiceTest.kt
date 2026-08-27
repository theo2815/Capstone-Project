package com.quickpitik.service.photographer

import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
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
import java.util.Optional
import java.util.UUID

// The async watermark pipeline: PROCESSING photos get their watermark
// derivative generated off-request and flip LIVE via a targeted conditional
// UPDATE. Attempt semantics mirror indexing's 2026-08-27 hardening — only
// semantic failures (undecodable bytes) burn processing_attempts; transport
// failures leave the budget intact for the reconcile sweep.
class PhotoWatermarkServiceTest {

    private lateinit var storageService: StorageService
    private lateinit var photoRepository: PhotoRepository
    private lateinit var settingsRepository: PhotographerSettingsRepository
    private lateinit var watermarkService: WatermarkService
    private lateinit var eventPublisher: ApplicationEventPublisher

    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private val photoId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        storageService = Mockito.mock(StorageService::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        watermarkService = Mockito.mock(WatermarkService::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
    }

    private fun service(maxAttempts: Int = 5) = PhotoWatermarkService(
        storageService,
        StorageProperties(),
        photoRepository,
        settingsRepository,
        watermarkService,
        // Real cache over the mocked storage — first get() always calls through.
        WatermarkLogoCache(storageService, SimpleMeterRegistry()),
        eventPublisher,
        TransactionTemplate(
            Mockito.mock(PlatformTransactionManager::class.java).also {
                Mockito.`when`(it.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
            },
        ),
        SimpleMeterRegistry(),
        maxAttempts,
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

    private fun stubHappyPath() {
        Mockito.`when`(photoRepository.findById(photoId)).thenReturn(Optional.of(photo()))
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(
            Optional.of(
                PhotographerSettings(
                    userId = photographerId,
                    watermarkS3Key = "watermarks/logo.png",
                    verificationStatus = VerificationStatus.APPROVED,
                ),
            ),
        )
        Mockito.`when`(storageService.getBytes("events/$eventId/photos/$photoId/original.jpg"))
            .thenReturn(byteArrayOf(1))
        Mockito.`when`(storageService.getBytes("watermarks/logo.png")).thenReturn(byteArrayOf(2))
        Mockito.`when`(watermarkService.processThumbnail(anyArg(), anyArg())).thenReturn(byteArrayOf(3))
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")
    }

    @Test
    fun `success stores the watermark, flips LIVE, and publishes photo-published`() {
        stubHappyPath()
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg())).thenReturn(1)

        service().process(photoId)

        val watermarkKey = "events/$eventId/photos/$photoId/watermark.jpg"
        Mockito.verify(storageService).put(watermarkKey, byteArrayOf(3), "image/jpeg")
        Mockito.verify(photoRepository).publishWatermarked(photoId, watermarkKey)
        Mockito.verify(eventPublisher).publishEvent(anyArg<PhotoPublishedEvent>())
        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
    }

    @Test
    fun `a lost flip race publishes no websocket frame`() {
        stubHappyPath()
        // Another worker (sweep vs hot path) flipped it first — 0 rows updated.
        Mockito.`when`(photoRepository.publishWatermarked(anyArg(), anyArg())).thenReturn(0)

        service().process(photoId)

        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
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
            .`when`(watermarkService).processThumbnail(anyArg(), anyArg())

        service().process(photoId)

        Mockito.verify(photoRepository).incrementProcessingAttempts(photoId)
        Mockito.verify(storageService, Mockito.never())
            .put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
        Mockito.verify(photoRepository, Mockito.never()).publishWatermarked(anyArg(), anyArg())
        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
    }

    @Test
    fun `a storage transport failure keeps the attempt budget intact`() {
        Mockito.`when`(photoRepository.findById(photoId)).thenReturn(Optional.of(photo()))
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(
            Optional.of(
                PhotographerSettings(
                    userId = photographerId,
                    watermarkS3Key = "watermarks/logo.png",
                    verificationStatus = VerificationStatus.APPROVED,
                ),
            ),
        )
        Mockito.`when`(storageService.getBytes(anyArg()))
            .thenThrow(RuntimeException("storage unreachable"))

        service().process(photoId)

        Mockito.verify(photoRepository, Mockito.never()).incrementProcessingAttempts(anyArg())
        Mockito.verify(photoRepository, Mockito.never()).publishWatermarked(anyArg(), anyArg())
        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg())
    }
}
