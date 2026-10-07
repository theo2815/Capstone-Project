package com.quickpitik.service.photos

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.FaceErasureResult
import com.quickpitik.dto.ai.JobStatusResult
import com.quickpitik.entity.*
import com.quickpitik.repository.PhotoFaceWork
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.*

class PhotoFaceErasureTest {
    private val event = UUID.randomUUID()
    private val photoId = UUID.randomUUID()
    private val person = UUID.randomUUID().toString()
    private val photos = mock(PhotoRepository::class.java)
    private val ai = mock(AiApiClient::class.java)
    private val rek = mock(RekognitionAiClient::class.java)
    private val fence = mock(PhotoWorkFence::class.java)
    @Suppress("UNCHECKED_CAST")
    private val adapters = mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
    private val manager = mock(PlatformTransactionManager::class.java)
    private val json = jacksonObjectMapper()
    private lateinit var service: PhotoFaceErasureService
    private val photo = Photo(id = photoId, eventId = event, s3Key = "synthetic", pricePhp = BigDecimal.TEN)
    private fun <T> anyArg(): T = org.mockito.ArgumentMatchers.any()
    private fun <T> eqArg(value: T): T = org.mockito.ArgumentMatchers.eq(value)

    @BeforeEach fun setup() {
        `when`(adapters.ifAvailable).thenReturn(rek)
        `when`(manager.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
        `when`(photos.findById(photoId)).thenReturn(Optional.of(photo))
        `when`(photos.findAllByIdForUpdate(listOf(photoId))).thenAnswer { photos.findById(photoId).map { listOf(it) }.orElse(emptyList()) }
        `when`(photos.faceWorkForPhoto(event, photoId)).thenReturn(emptyList())
        `when`(ai.listPersonsForEvent(event)).thenReturn(listOf(AiPersonRef(person, OffsetDateTime.now(), photoId.toString())))
        service = create(true)
    }

    private fun create(enabled: Boolean) = PhotoFaceErasureService(photos, ai, adapters,
        AiApiProperties(enabled = enabled), fence, json, TransactionTemplate(manager), Clock.systemUTC())

    @Test fun `providerChangeCannotHideOldEmbeddings`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person, FaceProvider.AI_API))
        val recordedProvider = FaceProvider.AI_API
        var cleanupAdapterUsed: FaceProvider? = null
        `when`(ai.erasePhotoFaces(event, person)).thenAnswer {
            cleanupAdapterUsed = FaceProvider.AI_API
            FaceErasureResult.DELETED
        }

        assertEquals(FaceErasureResult.DELETED, service.erase(event, photoId))

        assertEquals(recordedProvider, cleanupAdapterUsed)
        verifyNoInteractions(rek)
    }

    @Test fun `lateEnrollmentStaysCleanupPending`() {
        val output = work(finished = false)
        `when`(photos.faceWorkForPhoto(event, photoId)).thenReturn(listOf(output))
        `when`(photos.hasUnfinishedWork(photoId, "INDEXING")).thenReturn(true)
        val unresolvedRemoteEnrollAllowsCompletion = runCatching { service.erase(event, photoId) }.isSuccess
        assertFalse(unresolvedRemoteEnrollAllowsCompletion)
        verify(fence, never()).finish(anyArg())
    }

    @Test fun `unknown legacy person uses exact photo name and includes duplicate enrollments`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person))
        val duplicate = UUID.randomUUID().toString()
        val unrelated = UUID.randomUUID().toString()
        `when`(ai.listPersonsForEvent(event)).thenReturn(listOf(
            AiPersonRef(person, OffsetDateTime.now(), photoId.toString()),
            AiPersonRef(duplicate, OffsetDateTime.now(), photoId.toString()),
            AiPersonRef(unrelated, OffsetDateTime.now(), UUID.randomUUID().toString())))
        `when`(ai.erasePhotoFaces(event, person)).thenReturn(FaceErasureResult.DELETED)
        `when`(ai.erasePhotoFaces(event, duplicate)).thenReturn(FaceErasureResult.DELETED)

        assertEquals(FaceErasureResult.DELETED, service.erase(event, photoId))
        verify(ai).erasePhotoFaces(event, duplicate)
        verify(ai, never()).erasePhotoFaces(event, unrelated)
    }

    @Test fun `unattributed legacy person stays visible for retry`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person))
        `when`(ai.listPersonsForEvent(event)).thenReturn(listOf(AiPersonRef(person, OffsetDateTime.now(), "someone-else")))
        assertFailsWith<AiApiException> { service.erase(event, photoId) }
        verify(ai, never()).erasePhotoFaces(anyArg(), anyArg())
    }

    @Test fun `repeated legacy cleanup survives original person disappearance`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person))
        doReturn(listOf(photo)).`when`(photos).findAllByIdForUpdate(listOf(photoId))
        var present = true
        `when`(ai.listPersonsForEvent(event)).thenAnswer {
            if (present) listOf(AiPersonRef(person, OffsetDateTime.now(), photoId.toString())) else emptyList()
        }
        `when`(ai.erasePhotoFaces(event, person)).thenAnswer {
            if (present) { present = false; FaceErasureResult.DELETED } else FaceErasureResult.ALREADY_ABSENT
        }
        assertEquals(FaceErasureResult.DELETED, service.erase(event, photoId))
        org.junit.jupiter.api.Assertions.assertAll(
            { assertEquals(FaceProvider.AI_API, photo.facePersons.single().provider) },
            { assertEquals(FaceErasureResult.ALREADY_ABSENT, create(true).erase(event, photoId)) },
        )
    }

    @Test fun `legacy duplicate partial failure can retry after original deletion`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person))
        doReturn(listOf(photo)).`when`(photos).findAllByIdForUpdate(listOf(photoId))
        val duplicate = UUID.randomUUID().toString()
        val remaining = linkedSetOf(person, duplicate)
        `when`(ai.listPersonsForEvent(event)).thenAnswer {
            remaining.map { AiPersonRef(it, OffsetDateTime.now(), photoId.toString()) }
        }
        `when`(ai.erasePhotoFaces(event, person)).thenAnswer {
            remaining.remove(person)
            FaceErasureResult.DELETED
        }
        var failDuplicate = true
        `when`(ai.erasePhotoFaces(event, duplicate)).thenAnswer {
            if (failDuplicate) {
                failDuplicate = false
                throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, "AI_API_UNAVAILABLE", "Synthetic retry")
            }
            remaining.remove(duplicate)
            FaceErasureResult.DELETED
        }
        assertFailsWith<AiApiException> { service.erase(event, photoId) }
        assertEquals(FaceErasureResult.DELETED, create(true).erase(event, photoId))
        assertTrue(remaining.isEmpty())
    }

    @Test fun `failed provenance transaction prevents destructive legacy cleanup`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person))
        `when`(manager.getTransaction(anyArg())).thenThrow(IllegalStateException("Synthetic persistence failure"))
        org.junit.jupiter.api.Assertions.assertAll(
            { assertFailsWith<AiApiException> { service.erasePrior(photo) } },
            { verify(ai, never()).erasePhotoFaces(anyArg(), anyArg()) },
        )
    }

    @Test fun `changed legacy target cannot be deleted without durable proof`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person))
        val changed = Photo(id = photoId, eventId = event, s3Key = "synthetic", pricePhp = BigDecimal.TEN)
        doReturn(listOf(changed)).`when`(photos).findAllByIdForUpdate(listOf(photoId))
        org.junit.jupiter.api.Assertions.assertAll(
            { assertFailsWith<AiApiException> { service.erasePrior(photo) } },
            { verify(ai, never()).erasePhotoFaces(anyArg(), anyArg()) },
        )
    }

    @Test fun `wrong ledger photo name cannot prove a vanished legacy local target`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person))
        val changed = Photo(id = photoId, eventId = event, s3Key = "synthetic", pricePhp = BigDecimal.TEN)
        doReturn(listOf(changed)).`when`(photos).findAllByIdForUpdate(listOf(photoId))
        val output = work(finished = true)
        `when`(output.personName).thenReturn(UUID.randomUUID().toString())
        `when`(photos.faceWorkForPhoto(event, photoId)).thenReturn(listOf(output))
        org.junit.jupiter.api.Assertions.assertAll(
            { assertFailsWith<AiApiException> { service.erasePrior(photo) } },
            { verify(ai, never()).erasePhotoFaces(anyArg(), anyArg()) },
        )
    }

    @Test fun `missing recorded adapter cannot be replaced by current provider`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, "$event.$photoId", FaceProvider.REKOGNITION))
        `when`(adapters.ifAvailable).thenReturn(null)
        val error = assertFailsWith<AiApiException> { service.erase(event, photoId) }
        assertEquals("AI_API_UNAVAILABLE", error.aiCode)
        verifyNoInteractions(ai)
    }

    @Test fun `disabled AI waits when an indexed target exists`() {
        photo.facePersons.add(PhotoFacePersonEmbed(0, person, FaceProvider.AI_API))
        assertFailsWith<AiApiException> { create(false).erase(event, photoId) }
        verifyNoInteractions(ai, rek)
    }

    @Test fun `never indexed photo can skip disabled provider`() {
        assertEquals(FaceErasureResult.ALREADY_ABSENT, create(false).erase(event, photoId))
        verifyNoInteractions(ai, rek)
    }

    @Test fun `attempted legacy photo without provenance cannot prove absence`() {
        photo.indexingAttempts = 1
        assertFalse(runCatching { service.erase(event, photoId) }.isSuccess)
    }

    @Test fun `unknown ledger provider with no local ids cannot certify absence`() {
        val output = work(finished = true)
        `when`(output.provider).thenReturn("unknown")
        `when`(output.personIds).thenReturn("[]")
        `when`(photos.faceWorkForPhoto(event, photoId)).thenReturn(listOf(output))
        `when`(photos.confirmFaceWorkErased(eqArg(output.token), eqArg("[]"), anyArg())).thenReturn(1)
        assertFalse(runCatching { service.erase(event, photoId) }.isSuccess)
    }

    @Test fun `a newly finished output during cleanup cannot be certified erased`() {
        val first = work(finished = true)
        val late = work(finished = true)
        val ledger = mutableListOf(first)
        `when`(photos.faceWorkForPhoto(event, photoId)).thenAnswer { ledger.toList() }
        `when`(photos.confirmFaceWorkErased(eqArg(first.token), eqArg("[\"$person\"]"), anyArg())).thenReturn(1)
        `when`(ai.erasePhotoFaces(event, person)).thenAnswer {
            ledger.add(late)
            FaceErasureResult.DELETED
        }
        assertFalse(runCatching { service.erase(event, photoId) }.isSuccess)
    }

    @Test fun `last photo cleanup reads durable targets without an event catalog row`() {
        `when`(photos.findById(photoId)).thenReturn(Optional.empty())
        val output = work(finished = true)
        `when`(photos.faceWorkForPhoto(event, photoId)).thenReturn(listOf(output))
        `when`(ai.erasePhotoFaces(event, person)).thenReturn(FaceErasureResult.DELETED)
        `when`(photos.confirmFaceWorkErased(eqArg(output.token), eqArg("[\"$person\"]"), anyArg())).thenReturn(1)
        assertEquals(FaceErasureResult.DELETED, service.erase(event, photoId))
        verify(ai).erasePhotoFaces(event, person)
    }

    @Test fun `terminal job late person is retained before restricted missing photo cleanup`() {
        `when`(photos.findById(photoId)).thenReturn(Optional.empty())
        val output = work(finished = false)
        `when`(output.personIds).thenReturn("[]")
        `when`(output.jobIds).thenReturn("{\"FACE\":\"face-job\",\"BIB\":\"bib-job\"}")
        `when`(output.batchPhotoIds).thenReturn("[\"$photoId\"]")
        `when`(photos.faceWorkForPhoto(event, photoId)).thenReturn(listOf(output))
        `when`(ai.jobStatusComplete("face-job")).thenReturn(JobStatusResult("face-job", "completed",
            result = listOf(mapOf("ref" to photoId.toString(), "person_id" to person))))
        `when`(ai.jobStatusComplete("bib-job")).thenReturn(JobStatusResult("bib-job", "completed", result = emptyList()))
        `when`(ai.erasePhotoFaces(event, person)).thenReturn(FaceErasureResult.DELETED)
        `when`(photos.confirmFaceWorkErased(eqArg(output.token), eqArg("[\"$person\"]"), anyArg())).thenReturn(1)

        assertEquals(FaceErasureResult.DELETED, service.erase(event, photoId))
        verify(fence).recordPerson(anyArg(), eqArg(person))
        verify(fence).finish(anyArg())
    }

    @Test fun `terminal ref without enrollment outcome cannot settle its writer`() {
        expectTerminalOutcome(mapOf("ref" to photoId.toString()))
        org.junit.jupiter.api.Assertions.assertAll(
            { assertFalse(runCatching { service.erase(event, photoId) }.isSuccess) },
            { verify(fence, never()).finish(anyArg()) },
            { verify(photos, never()).confirmFaceWorkErased(anyArg(), anyArg(), anyArg()) },
        )
    }

    @Test fun `malformed terminal person with blank error cannot settle its writer`() {
        expectTerminalOutcome(mapOf("ref" to photoId.toString(), "person_id" to 123, "error" to " "))
        org.junit.jupiter.api.Assertions.assertAll(
            { assertFalse(runCatching { service.erase(event, photoId) }.isSuccess) },
            { verify(fence, never()).finish(anyArg()) },
            { verify(photos, never()).confirmFaceWorkErased(anyArg(), anyArg(), anyArg()) },
        )
    }

    @Test fun `valid no face and per input error outcomes still settle completed writers`() {
        val otherPhoto = UUID.randomUUID()
        val output = expectTerminalOutcome(mapOf("ref" to photoId.toString(), "person_id" to null))
        `when`(output.batchPhotoIds).thenReturn("[\"$photoId\",\"$otherPhoto\"]")
        `when`(ai.jobStatusComplete("face-job")).thenReturn(JobStatusResult("face-job", "completed",
            result = listOf(mapOf("ref" to photoId.toString(), "person_id" to null),
                mapOf("ref" to otherPhoto.toString(), "error" to "Synthetic decode failure"))))
        assertEquals(FaceErasureResult.ALREADY_ABSENT, service.erase(event, photoId))
        verify(fence).finish(anyArg())
    }

    @Test fun `nonstring explicit terminal ref cannot settle its writer`() {
        val output = expectTerminalOutcome(mapOf("ref" to 17, "person_id" to person))
        `when`(photos.confirmFaceWorkErased(eqArg(output.token), eqArg("[\"$person\"]"), anyArg())).thenReturn(1)
        `when`(ai.erasePhotoFaces(event, person)).thenReturn(FaceErasureResult.DELETED)
        org.junit.jupiter.api.Assertions.assertAll(
            { assertFalse(runCatching { service.erase(event, photoId) }.isSuccess) },
            { verify(fence, never()).recordPerson(anyArg(), anyArg()) },
            { verify(fence, never()).finish(anyArg()) },
        )
    }

    @Test fun `null terminal ref preserves valid position fallback`() {
        expectTerminalOutcome(mapOf("ref" to null, "person_id" to null))
        assertEquals(FaceErasureResult.ALREADY_ABSENT, service.erase(event, photoId))
        verify(fence).finish(anyArg())
    }

    @Test fun `fractional terminal index cannot settle its writer`() {
        val output = expectTerminalOutcome(mapOf("index" to 0.5, "person_id" to person))
        `when`(photos.confirmFaceWorkErased(eqArg(output.token), eqArg("[\"$person\"]"), anyArg())).thenReturn(1)
        `when`(ai.erasePhotoFaces(event, person)).thenReturn(FaceErasureResult.DELETED)
        org.junit.jupiter.api.Assertions.assertAll(
            { assertFalse(runCatching { service.erase(event, photoId) }.isSuccess) },
            { verify(fence, never()).recordPerson(anyArg(), anyArg()) },
            { verify(fence, never()).finish(anyArg()) },
        )
    }

    @Test fun `null terminal index preserves valid position fallback`() {
        expectTerminalOutcome(mapOf("ref" to null, "index" to null, "person_id" to null))
        assertEquals(FaceErasureResult.ALREADY_ABSENT, service.erase(event, photoId))
        verify(fence).finish(anyArg())
    }

    private fun expectTerminalOutcome(result: Map<String, Any?>): PhotoFaceWork {
        val output = work(finished = false)
        `when`(output.personIds).thenReturn("[]")
        `when`(output.jobIds).thenReturn("{\"FACE\":\"face-job\",\"BIB\":\"bib-job\"}")
        `when`(output.batchPhotoIds).thenReturn("[\"$photoId\"]")
        `when`(photos.faceWorkForPhoto(event, photoId)).thenReturn(listOf(output))
        `when`(ai.listPersonsForEvent(event)).thenReturn(emptyList())
        `when`(ai.jobStatusComplete("face-job")).thenReturn(JobStatusResult("face-job", "completed", result = listOf(result)))
        `when`(ai.jobStatusComplete("bib-job")).thenReturn(JobStatusResult("bib-job", "completed", result = emptyList()))
        `when`(photos.confirmFaceWorkErased(eqArg(output.token), eqArg("[]"), anyArg())).thenReturn(1)
        return output
    }

    private fun work(finished: Boolean): PhotoFaceWork = mock(PhotoFaceWork::class.java).also {
        `when`(it.token).thenReturn(UUID.randomUUID())
        `when`(it.photoId).thenReturn(photoId)
        `when`(it.eventId).thenReturn(event)
        `when`(it.generation).thenReturn(0L)
        `when`(it.expiresAt).thenReturn(Instant.parse("2020-01-01T00:00:00Z"))
        `when`(it.provider).thenReturn("ai_api")
        `when`(it.personName).thenReturn(photoId.toString())
        `when`(it.personIds).thenReturn("[\"$person\"]")
        `when`(it.priorPersonIds).thenReturn("[]")
        `when`(it.jobIds).thenReturn("{}")
        `when`(it.batchPhotoIds).thenReturn("[]")
        `when`(it.finishedAt).thenReturn(if (finished) Instant.now() else null)
    }
}
