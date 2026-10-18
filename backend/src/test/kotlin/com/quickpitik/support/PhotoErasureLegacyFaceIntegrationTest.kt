package com.quickpitik.support

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.config.RekognitionProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photos.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.photos.*
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Assertions.assertAll
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.beans.factory.support.StaticListableBeanFactory
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.DeleteFacesRequest
import software.amazon.awssdk.services.rekognition.model.DeleteFacesResponse
import software.amazon.awssdk.services.rekognition.model.Face
import software.amazon.awssdk.services.rekognition.model.ListFacesRequest
import software.amazon.awssdk.services.rekognition.model.ListFacesResponse
import java.math.BigDecimal
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.*

// F1 shipped because request() and attempt() were only tested apart (audit 2026-10-18).
// This runs the real chain, with real SQL, on photos face-indexed before V75.
class PhotoErasureLegacyFaceIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var coverage: EventPhotographerRepository
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var jobs: PhotoErasureJobRepository
    @Autowired lateinit var intents: PhotoUploadIntentRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var grants: DownloadGrantRepository
    @Autowired lateinit var metadata: PhotoMetadataRepository
    @Autowired lateinit var availability: PhotoAvailabilityService
    @Autowired lateinit var fence: PhotoWorkFence
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var json: com.fasterxml.jackson.databind.ObjectMapper
    @Autowired lateinit var removalProperties: PhotoRemovalProperties
    @Autowired lateinit var storageProperties: StorageProperties

    private val sdk = Mockito.mock(RekognitionClient::class.java)
    private val ai = Mockito.mock(AiApiClient::class.java)
    private val storage = Mockito.mock(StorageService::class.java)
    private val faces = mutableSetOf<String>() // face ids still in the fake collection
    private lateinit var owner: User
    private lateinit var admin: User
    private lateinit var event: Event
    private lateinit var photo: Photo
    private lateinit var eraser: PhotoErasureService

    private fun externalId() = "${event.id}.${photo.id}"

    @BeforeEach fun seed() {
        owner = users.saveAndFlush(User(email = "legacy-face-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Owner", role = Role.PHOTOGRAPHER))
        admin = users.saveAndFlush(User(email = "legacy-face-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Admin", role = Role.ADMIN))
        event = events.saveAndFlush(Event(slug = "legacy-face-${UUID.randomUUID()}", name = "Legacy faces", date = LocalDate.now(),
            location = "Cebu", status = EventStatus.ACTIVE, photoCount = 1))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "legacy/${UUID.randomUUID()}/original", pricePhp = BigDecimal("125")))
        coverage.saveAndFlush(EventPhotographer(EventPhotographerId(event.id, owner.id), photoCount = 1, firstUploadAt = OffsetDateTime.now()))
        // Indexed by Rekognition before V75 added face provenance.
        jdbc.update("UPDATE photos SET indexed_provider = 'rekognition', indexing_attempts = 1 WHERE id = ?", photo.id)

        Mockito.`when`(sdk.listFaces(Mockito.any(ListFacesRequest::class.java))).thenAnswer {
            ListFacesResponse.builder().faces(faces.map { id -> Face.builder().faceId(id).externalImageId(externalId()).build() }).build()
        }
        Mockito.`when`(sdk.deleteFaces(Mockito.any(DeleteFacesRequest::class.java))).thenAnswer {
            val request = it.getArgument<DeleteFacesRequest>(0)
            faces.removeAll(request.faceIds().toSet())
            DeleteFacesResponse.builder().deletedFaces(request.faceIds()).build()
        }
        val rekognition = RekognitionAiClient(sdk, RekognitionProperties())
        val faceEraser = PhotoFaceErasureService(photos, ai,
            StaticListableBeanFactory(mapOf("rekognition" to rekognition)).getBeanProvider(RekognitionAiClient::class.java),
            AiApiProperties(enabled = true), fence, json, tx, Clock.systemUTC())
        eraser = PhotoErasureService(photos, jobs, intents, items, orderItems, grants, coverage, events, metadata,
            storage, faceEraser, removalProperties, storageProperties, tx, json, Clock.systemUTC(),
            io.micrometer.core.instrument.simple.SimpleMeterRegistry())
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

    @Test fun `privacy removal of a photo indexed before V75 erases its face and completes`() {
        // No provider column value: the V75 default makes it UNKNOWN, exactly as on production.
        jdbc.update("INSERT INTO photo_face_persons(photo_id, face_index, ai_person_id) VALUES (?, 0, ?)", photo.id, externalId())
        faces.add("face-1")

        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))

        assertAll(
            { assertTrue(faces.isEmpty(), "the face must be gone from the collection") },
            { assertNotNull(photos.findById(photo.id).orElseThrow().purgedAt) },
            { assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM photo_face_persons WHERE photo_id = ?", Int::class.java, photo.id)) },
            { assertEquals("rekognition", jdbc.queryForObject(
                "SELECT provider FROM photo_work_outputs WHERE photo_id = ? AND kind = 'INDEXING'", String::class.java, photo.id)) },
            { assertEquals(1, jdbc.queryForObject(
                "SELECT count(*) FROM photo_work_outputs WHERE photo_id = ? AND faces_erased_at IS NOT NULL", Int::class.java, photo.id)) },
        )
        Mockito.verify(sdk).deleteFaces(Mockito.argThat<DeleteFacesRequest> {
            it.collectionId() == "qp-event-${event.id}" && it.faceIds() == listOf("face-1")
        })
        Mockito.verifyNoInteractions(ai)
    }

    @Test fun `photo indexed before V75 with no face record completes after a Rekognition listing`() {
        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))
        assertNotNull(photos.findById(photo.id).orElseThrow().purgedAt)
        Mockito.verify(sdk, Mockito.atLeastOnce()).listFaces(Mockito.argThat<ListFacesRequest> { it.collectionId() == "qp-event-${event.id}" })
        Mockito.verifyNoInteractions(ai)
    }

    @Test fun `photo whose Rekognition indexing call ended without an answer completes erasure`() {
        // A throttled or dropped IndexFaces leaves its record open (PhotoIndexingService skips finish).
        val token = UUID.randomUUID()
        val expired = OffsetDateTime.now().minusMinutes(10)
        tx.executeWithoutResult {
            photos.recordWork(token, photo.id, event.id, 0, "INDEXING", "rekognition", expired, null, photo.id.toString(), null, "[]")
            assertEquals(1, photos.claimWork(photo.id, "INDEXING", token, expired))
        }
        faces.add("late-face") // the dropped call may still have written a face

        val job = eraser.request(privacy())
        assertEquals(PhotoErasureOutcome.COMPLETE, eraser.attempt(job))

        assertTrue(faces.isEmpty())
        assertNotNull(photos.findById(photo.id).orElseThrow().purgedAt)
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE token = ? " +
            "AND finished_at IS NOT NULL AND faces_erased_at IS NOT NULL", Int::class.java, token))
    }
}
