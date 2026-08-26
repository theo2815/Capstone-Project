package com.quickpitik.service.ai

import com.quickpitik.config.RekognitionProperties
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.ArgumentCaptor
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito.mock
import org.mockito.Mockito.verify
import org.mockito.Mockito.`when`
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.BoundingBox
import software.amazon.awssdk.services.rekognition.model.DeleteCollectionRequest
import software.amazon.awssdk.services.rekognition.model.DeleteCollectionResponse
import software.amazon.awssdk.services.rekognition.model.DetectFacesRequest
import software.amazon.awssdk.services.rekognition.model.DetectFacesResponse
import software.amazon.awssdk.services.rekognition.model.DetectTextRequest
import software.amazon.awssdk.services.rekognition.model.DetectTextResponse
import software.amazon.awssdk.services.rekognition.model.Face
import software.amazon.awssdk.services.rekognition.model.FaceDetail
import software.amazon.awssdk.services.rekognition.model.FaceMatch
import software.amazon.awssdk.services.rekognition.model.FaceRecord
import software.amazon.awssdk.services.rekognition.model.IndexFacesRequest
import software.amazon.awssdk.services.rekognition.model.IndexFacesResponse
import software.amazon.awssdk.services.rekognition.model.ResourceNotFoundException
import software.amazon.awssdk.services.rekognition.model.SearchFacesByImageRequest
import software.amazon.awssdk.services.rekognition.model.SearchFacesByImageResponse
import software.amazon.awssdk.services.rekognition.model.TextDetection
import software.amazon.awssdk.services.rekognition.model.TextTypes
import java.util.UUID

// Unit tests for the Rekognition → DTO mapping. No AWS network: RekognitionClient
// is mocked. Small byte arrays skip the downscaler (passthrough under 4.5 MB).
class RekognitionAiClientTest {
    private lateinit var rek: RekognitionClient
    private lateinit var client: RekognitionAiClient
    private val props = RekognitionProperties()
    private val eventId: UUID = UUID.randomUUID()
    private val photoId: UUID = UUID.randomUUID()
    private val bytes = ByteArray(1024) { 1 }

    @BeforeEach
    fun setup() {
        rek = mock(RekognitionClient::class.java)
        client = RekognitionAiClient(rek, props)
    }

    @Test
    fun `enroll returns the composite external id and the face count`() {
        `when`(rek.indexFaces(any(IndexFacesRequest::class.java))).thenReturn(
            IndexFacesResponse.builder().faceRecords(
                FaceRecord.builder().face(Face.builder().faceId("f1").build()).build(),
                FaceRecord.builder().face(Face.builder().faceId("f2").build()).build(),
            ).build(),
        )
        val result = client.facesEnroll(bytes, "image/jpeg", "$photoId.jpg", photoId.toString(), null, eventId)
        assertEquals("$eventId.$photoId", result.person_id)
        assertEquals(2, result.faces_enrolled)
    }

    @Test
    fun `enroll with zero faces raises NO_FACES (benign to the caller)`() {
        `when`(rek.indexFaces(any(IndexFacesRequest::class.java)))
            .thenReturn(IndexFacesResponse.builder().faceRecords(emptyList()).build())
        val ex = assertThrows<AiApiException> {
            client.facesEnroll(bytes, "image/jpeg", "x.jpg", photoId.toString(), null, eventId)
        }
        assertEquals("NO_FACES", ex.aiCode)
    }

    @Test
    fun `search dedupes by external id, normalizes similarity, and never searches below the floor`() {
        `when`(rek.searchFacesByImage(any(SearchFacesByImageRequest::class.java))).thenReturn(
            SearchFacesByImageResponse.builder().faceMatches(
                FaceMatch.builder().similarity(95f).face(Face.builder().externalImageId("$eventId.$photoId").build()).build(),
                FaceMatch.builder().similarity(88f).face(Face.builder().externalImageId("$eventId.$photoId").build()).build(),
            ).build(),
        )
        // Caller passes 0.6, but the provider floors at props.faceMatchThreshold (0.8).
        val result = client.facesSearch(bytes, "image/jpeg", "s.jpg", eventId, threshold = 0.6, topK = 50)

        assertEquals(1, result.matches.size)
        assertEquals("$eventId.$photoId", result.matches[0].person_id)
        assertEquals(0.95, result.matches[0].similarity, 1e-6)

        val cap = ArgumentCaptor.forClass(SearchFacesByImageRequest::class.java)
        verify(rek).searchFacesByImage(cap.capture())
        assertEquals(80.0f, cap.value.faceMatchThreshold())
    }

    @Test
    fun `search returns empty when the collection does not exist yet`() {
        `when`(rek.searchFacesByImage(any(SearchFacesByImageRequest::class.java)))
            .thenThrow(ResourceNotFoundException.builder().message("no collection").build())
        val result = client.facesSearch(bytes, "image/jpeg", "s.jpg", eventId, 0.8, 50)
        assertTrue(result.matches.isEmpty())
    }

    @Test
    fun `bib keeps numeric words at or above min chars and normalizes confidence`() {
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenReturn(
            DetectTextResponse.builder().textDetections(
                TextDetection.builder().type(TextTypes.WORD).detectedText("1234").confidence(99f).build(),
                TextDetection.builder().type(TextTypes.WORD).detectedText("SPONSOR").confidence(97f).build(),
                TextDetection.builder().type(TextTypes.LINE).detectedText("1234").confidence(99f).build(),
                TextDetection.builder().type(TextTypes.WORD).detectedText("7").confidence(90f).build(),
            ).build(),
        )
        val result = client.bibsRecognize(bytes, "image/jpeg", "p.jpg", minChars = 2)
        assertEquals(1, result.detections.size)
        assertEquals("1234", result.detections[0].bib_number)
        assertEquals(0.99, result.detections[0].confidence, 1e-6)
    }

    @Test
    fun `detect faces maps count and normalizes confidence`() {
        `when`(rek.detectFaces(any(DetectFacesRequest::class.java))).thenReturn(
            DetectFacesResponse.builder().faceDetails(
                FaceDetail.builder().confidence(99.5f)
                    .boundingBox(BoundingBox.builder().left(0.1f).top(0.1f).width(0.2f).height(0.2f).build())
                    .build(),
            ).build(),
        )
        val result = client.facesDetect(bytes, "image/jpeg", "s.jpg")
        assertEquals(1, result.faces.size)
        assertEquals(0.995, result.faces[0].confidence, 1e-6)
    }

    @Test
    fun `delete-by-event drops the event's collection`() {
        `when`(rek.deleteCollection(any(DeleteCollectionRequest::class.java)))
            .thenReturn(DeleteCollectionResponse.builder().statusCode(200).build())
        client.deleteFacesByEvent(eventId)
        val cap = ArgumentCaptor.forClass(DeleteCollectionRequest::class.java)
        verify(rek).deleteCollection(cap.capture())
        assertEquals("qp-event-$eventId", cap.value.collectionId())
    }
}
