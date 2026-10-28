package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IndexingMode
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.repository.PhotoRepository
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertTrue

// Reconciliation/hot-path trigger: mode-gating (BATCH stands the per-photo path
// down) and the retryable-status selection — including the #3 fix that makes
// PARTIAL self-healing.
class PhotoIndexingTriggerTest {

    private lateinit var indexingService: PhotoIndexingService
    private lateinit var photoRepo: PhotoRepository
    private val registry = SimpleMeterRegistry()

    // Mockito.any() returns null at runtime; declared as a (platform) T so Kotlin
    // inserts no null-assertion — usable as a matcher for non-null params.
    private fun <T> anyArg(): T = Mockito.any()

    // Same trick for ArgumentCaptor.capture(): the declared T return lets Kotlin
    // pass the (null) result to a non-null Kotlin parameter without a checkNotNull.
    private fun <T> capture(c: ArgumentCaptor<T>): T = c.capture()

    @BeforeEach
    fun setUp() {
        indexingService = Mockito.mock(PhotoIndexingService::class.java)
        photoRepo = Mockito.mock(PhotoRepository::class.java)
    }

    private fun trigger(props: AiApiProperties) = PhotoIndexingTrigger(indexingService, photoRepo, props, registry)

    // FS-2 (R5-1): every photo the sweep picks up is one the hot path missed.
    @Test
    fun `reconcile counts the photos it picks up`() {
        Mockito.`when`(photoRepo.findIndexingBacklog(anyArg(), Mockito.anyInt(), anyArg(), anyArg()))
            .thenReturn(listOf(photo(), photo()), emptyList())

        trigger(AiApiProperties(enabled = true)).reconcile()
        trigger(AiApiProperties(enabled = true)).reconcile()

        kotlin.test.assertEquals(2.0, registry.counter("qp.indexing.reconciled").count())
    }

    private fun photo() = Photo(eventId = UUID.randomUUID(), s3Key = "k", pricePhp = BigDecimal.TEN)

    // FS-3: photos the hot path dropped are picked up 30 s after upload, not 120 s.
    @Test
    fun `reconcile picks up photos uploaded more than 30 seconds ago`() {
        var cutoff: OffsetDateTime? = null
        Mockito.`when`(photoRepo.findIndexingBacklog(anyArg(), Mockito.anyInt(), anyArg(), anyArg()))
            .thenAnswer { cutoff = it.getArgument(2); emptyList<Photo>() }
        val before = OffsetDateTime.now()

        trigger(AiApiProperties(enabled = true)).reconcile()

        val after = OffsetDateTime.now()
        assertTrue(!cutoff!!.isBefore(before.minusSeconds(30)) && !cutoff!!.isAfter(after.minusSeconds(30)), "cutoff $cutoff")
    }

    @Test
    fun `reconcile in BATCH mode does nothing`() {
        trigger(AiApiProperties(enabled = true, indexingMode = IndexingMode.BATCH)).reconcile()

        Mockito.verifyNoInteractions(photoRepo)
        Mockito.verifyNoInteractions(indexingService)
    }

    @Test
    fun `reconcile when ai-api disabled does nothing`() {
        trigger(AiApiProperties(enabled = false)).reconcile()

        Mockito.verifyNoInteractions(photoRepo)
        Mockito.verifyNoInteractions(indexingService)
    }

    @Test
    fun `reconcile re-drives each backlog photo and selects PENDING, FAILED, PARTIAL`() {
        val p1 = photo()
        val p2 = photo()
        Mockito.`when`(photoRepo.findIndexingBacklog(anyArg(), Mockito.anyInt(), anyArg(), anyArg()))
            .thenReturn(listOf(p1, p2))

        trigger(AiApiProperties(enabled = true)).reconcile()

        Mockito.verify(indexingService).index(p1.id)
        Mockito.verify(indexingService).index(p2.id)

        @Suppress("UNCHECKED_CAST")
        val statuses =
            ArgumentCaptor.forClass(Collection::class.java) as ArgumentCaptor<Collection<IndexingStatus>>
        Mockito.verify(photoRepo).findIndexingBacklog(
            capture(statuses),
            Mockito.anyInt(),
            anyArg(),
            anyArg(),
        )
        // #3: PARTIAL is retryable so a transiently-failed half self-heals.
        assertTrue(
            statuses.value.containsAll(
                listOf(IndexingStatus.PENDING, IndexingStatus.FAILED, IndexingStatus.PARTIAL),
            ),
        )
    }

    @Test
    fun `onPhotoUploaded in BATCH mode does not index on the hot path`() {
        trigger(AiApiProperties(enabled = true, indexingMode = IndexingMode.BATCH))
            .onPhotoUploaded(PhotoUploadedForIndexing(UUID.randomUUID(), UUID.randomUUID()))

        Mockito.verifyNoInteractions(indexingService)
    }

    @Test
    fun `onPhotoUploaded in PER_PHOTO mode indexes the uploaded photo`() {
        val photoId = UUID.randomUUID()
        trigger(AiApiProperties(enabled = true))
            .onPhotoUploaded(PhotoUploadedForIndexing(photoId, UUID.randomUUID()))

        Mockito.verify(indexingService).index(photoId)
    }
}
