package com.quickpitik.support

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
import com.quickpitik.config.AiProvider
import com.quickpitik.dto.ai.BibDetection
import com.quickpitik.dto.ai.BibsRecognizeResult
import com.quickpitik.dto.ai.FacesEnrollResult
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.events.EventFaceExpiry
import com.quickpitik.service.photographer.PhotoIndexingService
import com.quickpitik.service.photos.PhotoFaceErasureService
import com.quickpitik.service.photos.PhotoWorkFence
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

// IX-W3a: a DetectText word longer than photo_bibs.bib_number (VARCHAR(20)) used to
// fail the outcome write and strand the photo's indexing token.
class PhotoIndexingBibLengthIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var fence: PhotoWorkFence
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var jdbc: JdbcTemplate

    private fun <T> anyArg(): T = Mockito.any()

    @Test fun `a photo with an over-long detected word is indexed and its work token is cleared`() {
        val owner = users.saveAndFlush(User(email = "bib-length-${UUID.randomUUID()}@test.local",
            passwordHash = "unused", name = "Photographer", role = Role.PHOTOGRAPHER))
        val event = events.saveAndFlush(Event(slug = "bib-length-${UUID.randomUUID()}", name = "Cebu run",
            date = LocalDate.of(2026, 10, 6), location = "Cebu", status = EventStatus.ACTIVE))
        val photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "synthetic/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("125")))

        val bytes = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0x00)
        val storage = Mockito.mock(StorageService::class.java)
        Mockito.`when`(storage.getBytes(photo.s3Key)).thenReturn(bytes)
        val provider = Mockito.mock(FaceBibProvider::class.java)
        Mockito.`when`(provider.prepareInlineImage(anyArg())).thenAnswer { it.getArgument<ByteArray>(0) }
        Mockito.`when`(provider.facesEnroll(bytes, "image/jpeg", "${photo.id}.jpg", photo.id.toString(), null, event.id))
            .thenReturn(FacesEnrollResult(person_id = "person-${photo.id}"))
        Mockito.`when`(provider.bibsRecognize(bytes, "image/jpeg", "${photo.id}.jpg")).thenReturn(BibsRecognizeResult(listOf(
            BibDetection("A".repeat(21), 0.95), BibDetection("1234", 0.92))))
        val indexer = PhotoIndexingService(photos, storage, provider, AiApiProperties(enabled = true),
            AiProperties(AiProvider.AI_API), Mockito.mock(ApplicationEventPublisher::class.java), tx, SimpleMeterRegistry(),
            fence, Mockito.mock(PhotoFaceErasureService::class.java), Mockito.mock(EventFaceExpiry::class.java))

        indexer.index(photo.id)

        val saved = photos.findById(photo.id).orElseThrow()
        assertEquals(IndexingStatus.INDEXED, saved.indexingStatus)
        assertNull(saved.indexingWorkToken)
        assertEquals(listOf("1234"), jdbc.queryForList(
            "SELECT bib_number FROM photo_bibs WHERE photo_id = ?", String::class.java, photo.id))
        assertEquals(0, jdbc.queryForObject(
            "SELECT count(*) FROM photo_work_outputs WHERE photo_id = ? AND finished_at IS NULL", Int::class.java, photo.id))
    }
}
