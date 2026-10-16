package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.repository.*
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.*

class PhotoFaceErasureIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var fence: com.quickpitik.service.photos.PhotoWorkFence

    private fun fixture(): Photo {
        val event = events.saveAndFlush(Event(slug = "faces-${UUID.randomUUID()}", name = "Cebu",
            date = LocalDate.of(2026, 10, 7), location = "Cebu", status = EventStatus.ACTIVE))
        return photos.saveAndFlush(Photo(eventId = event.id, s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal.TEN))
    }

    private fun output(photo: Photo, finished: Boolean = true): UUID {
        val token = UUID.randomUUID()
        tx.executeWithoutResult {
            photos.recordWork(token, photo.id, photo.eventId, 0, "INDEXING", "ai_api", OffsetDateTime.now(),
                null, photo.id.toString(), "rekognition", "[\"${photo.eventId}.${photo.id}\"]")
            photos.recordWorkPersons(token, "[\"person-a\",\"person-b\"]")
            photos.recordWorkJob(token, "{\"FACE\":\"job-a\"}")
            if (finished) photos.finishWorkOutput(token, OffsetDateTime.now())
        }
        return token
    }

    @Test fun `event scoped provenance survives deletion of the last photo`() {
        val photo = fixture()
        val token = output(photo)
        photos.deleteById(photo.id)
        val work = photos.faceWorkForPhoto(photo.eventId, photo.id).single()
        assertEquals(token, work.token)
        assertEquals("ai_api", work.provider)
        assertEquals(photo.id.toString(), work.personName)
        assertEquals("[\"person-a\", \"person-b\"]", work.personIds)
        assertEquals("rekognition", work.priorProvider)
        assertNotNull(work.finishedAt)
        assertTrue(photos.faceWorkForPhoto(UUID.randomUUID(), photo.id).isEmpty())
        assertTrue(photos.faceWorkForPhoto(photo.eventId, UUID.randomUUID()).isEmpty())
        assertTrue(photos.faceWorkPending(PageRequest.of(0, 100)).any { it.photoId == photo.id && it.eventId == photo.eventId })
    }

    @Test fun `absence confirmation requires finished unchanged targets`() {
        val photo = fixture()
        val token = output(photo, finished = false)
        tx.executeWithoutResult {
            assertEquals(0, photos.confirmFaceWorkErased(token, "[\"person-a\",\"person-b\"]", OffsetDateTime.now()))
            photos.finishWorkOutput(token, OffsetDateTime.now())
            assertEquals(0, photos.confirmFaceWorkErased(token, "[\"person-a\"]", OffsetDateTime.now()))
            assertEquals(1, photos.confirmFaceWorkErased(token, "[\"person-a\",\"person-b\"]", OffsetDateTime.now()))
        }
        assertNotNull(photos.faceWorkForPhoto(photo.eventId, photo.id).single().facesErasedAt)
        tx.executeWithoutResult { photos.recordWorkPersons(token, "[\"late-person\"]") }
        assertNull(photos.faceWorkForPhoto(photo.eventId, photo.id).single().facesErasedAt)
    }

    @Test fun `legacy provenance remains unknown while new face provenance round trips`() {
        val photo = fixture()
        jdbc.update("INSERT INTO photo_face_persons(photo_id,face_index,ai_person_id) VALUES (?,?,?)", photo.id, 0, "legacy")
        assertEquals(FaceProvider.UNKNOWN, photos.findById(photo.id).orElseThrow().facePersons.single().provider)
        tx.executeWithoutResult {
            photos.findById(photo.id).orElseThrow().facePersons.apply {
                clear()
                add(PhotoFacePersonEmbed(0, "new-person", FaceProvider.AI_API))
            }
        }
        assertEquals(FaceProvider.AI_API, photos.findById(photo.id).orElseThrow().facePersons.single().provider)
    }

    @Test fun `pending face discovery ignores ordinary live indexed photos`() {
        val photo = fixture()
        output(photo)
        assertFalse(photos.faceWorkPending(PageRequest.of(0, 100)).any { it.photoId == photo.id })
        photos.deleteById(photo.id)
        assertTrue(photos.faceWorkPending(PageRequest.of(0, 100)).any { it.photoId == photo.id })
    }

    @Test fun `legacy work attribution requires unchanged scoped whole group`() {
        val photo = fixture()
        val token = UUID.randomUUID()
        val wrongName = UUID.randomUUID()
        tx.executeWithoutResult {
            photos.recordWork(token, photo.id, photo.eventId, 0, "INDEXING", "unknown", OffsetDateTime.now(),
                null, photo.id.toString(), null, "[\"prior-a\",\"prior-b\"]")
            photos.recordWorkPersons(token, "[\"current-a\"]")
            photos.recordWork(wrongName, photo.id, photo.eventId, 0, "INDEXING", "unknown", OffsetDateTime.now(),
                null, "different-photo", null, "[\"prior-a\"]")
            assertEquals(0, photos.attributeFaceWorkProvider(wrongName, photo.eventId, photo.id, true, "[\"prior-a\"]", "ai_api"))
            assertEquals(0, photos.attributeFaceWorkProvider(token, UUID.randomUUID(), photo.id, false, "[\"current-a\"]", "ai_api"))
            assertEquals(0, photos.attributeFaceWorkProvider(token, photo.eventId, UUID.randomUUID(), false, "[\"current-a\"]", "ai_api"))
            assertEquals(0, photos.attributeFaceWorkProvider(token, photo.eventId, photo.id, true, "[\"prior-a\"]", "ai_api"))
            assertEquals(1, photos.attributeFaceWorkProvider(token, photo.eventId, photo.id, false, "[\"current-a\"]", "ai_api"))
            assertEquals(1, photos.attributeFaceWorkProvider(token, photo.eventId, photo.id, true, "[\"prior-a\",\"prior-b\"]", "ai_api"))
            assertEquals(0, photos.attributeFaceWorkProvider(token, photo.eventId, photo.id, true, "[\"prior-a\",\"prior-b\"]", "rekognition"))
        }
        val work = photos.faceWorkForPhoto(photo.eventId, photo.id).single { it.token == token }
        assertEquals("ai_api", work.provider)
        assertEquals("ai_api", work.priorProvider)
    }

    @Test fun `legacy embedded and prior proof commits before deletion and survives missing photo`() {
        val photo = fixture()
        photo.facePersons.add(PhotoFacePersonEmbed(0, "legacy-person"))
        photos.saveAndFlush(photo)
        val token = UUID.randomUUID()
        tx.executeWithoutResult {
            photos.recordWork(token, photo.id, photo.eventId, 0, "INDEXING", "ai_api", OffsetDateTime.now(),
                null, photo.id.toString(), null, "[\"legacy-person\"]")
            photos.finishWorkOutput(token, OffsetDateTime.now())
        }
        val ai = org.mockito.Mockito.mock(com.quickpitik.service.ai.AiApiClient::class.java)
        var present = true
        org.mockito.Mockito.`when`(ai.listPersonsForEvent(photo.eventId)).thenAnswer {
            if (present) listOf(com.quickpitik.service.ai.AiPersonRef("legacy-person", OffsetDateTime.now(), photo.id.toString())) else emptyList()
        }
        org.mockito.Mockito.`when`(ai.erasePhotoFaces(photo.eventId, "legacy-person")).thenAnswer {
            assertFalse(org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive())
            assertEquals("ai_api", photos.faceWorkForPhoto(photo.eventId, photo.id).single().priorProvider)
            if (present) {
                assertEquals(FaceProvider.AI_API, photos.findById(photo.id).orElseThrow().facePersons.single().provider)
                present = false
                com.quickpitik.dto.ai.FaceErasureResult.DELETED
            } else com.quickpitik.dto.ai.FaceErasureResult.ALREADY_ABSENT
        }
        val service = com.quickpitik.service.photos.PhotoFaceErasureService(photos, ai,
            org.springframework.beans.factory.support.StaticListableBeanFactory().getBeanProvider(com.quickpitik.service.ai.RekognitionAiClient::class.java),
            com.quickpitik.config.AiApiProperties(enabled = true), fence,
            com.fasterxml.jackson.module.kotlin.jacksonObjectMapper(), tx, java.time.Clock.systemUTC())
        service.erasePrior(photo)
        photos.deleteById(photo.id)
        assertEquals(com.quickpitik.dto.ai.FaceErasureResult.ALREADY_ABSENT, service.erase(photo.eventId, photo.id))
        assertNotNull(photos.faceWorkForPhoto(photo.eventId, photo.id).single().facesErasedAt)
    }

    @Test fun `unattributed sibling rolls back embedded and ledger proof before deletion`() {
        val photo = fixture()
        photo.facePersons.add(PhotoFacePersonEmbed(0, "legacy-person"))
        photos.saveAndFlush(photo)
        val first = UUID.randomUUID()
        tx.executeWithoutResult {
            photos.recordWork(first, photo.id, photo.eventId, 0, "INDEXING", "ai_api", OffsetDateTime.now(),
                null, photo.id.toString(), null, "[\"legacy-person\"]")
        }
        tx.executeWithoutResult {
            photos.recordWork(UUID.randomUUID(), photo.id, photo.eventId, 0, "INDEXING", "ai_api", OffsetDateTime.now(),
                null, photo.id.toString(), null, "[\"legacy-person\",\"unattributed-person\"]")
        }
        val ai = org.mockito.Mockito.mock(com.quickpitik.service.ai.AiApiClient::class.java)
        org.mockito.Mockito.`when`(ai.listPersonsForEvent(photo.eventId)).thenReturn(listOf(
            com.quickpitik.service.ai.AiPersonRef("legacy-person", OffsetDateTime.now(), photo.id.toString())))
        val service = com.quickpitik.service.photos.PhotoFaceErasureService(photos, ai,
            org.springframework.beans.factory.support.StaticListableBeanFactory().getBeanProvider(com.quickpitik.service.ai.RekognitionAiClient::class.java),
            com.quickpitik.config.AiApiProperties(enabled = true), fence,
            com.fasterxml.jackson.module.kotlin.jacksonObjectMapper(), tx, java.time.Clock.systemUTC())
        assertFailsWith<com.quickpitik.service.ai.AiApiException> { service.erasePrior(photo) }
        assertEquals(FaceProvider.UNKNOWN, photos.findById(photo.id).orElseThrow().facePersons.single().provider)
        assertTrue(photos.faceWorkForPhoto(photo.eventId, photo.id).all { it.priorProvider == null })
        org.mockito.Mockito.verify(ai, org.mockito.Mockito.never()).erasePhotoFaces(photo.eventId, "legacy-person")
    }
}
