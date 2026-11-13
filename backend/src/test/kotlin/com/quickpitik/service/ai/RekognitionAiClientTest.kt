package com.quickpitik.service.ai

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.read.ListAppender
import com.quickpitik.config.RekognitionProperties
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.ArgumentCaptor
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito.mock
import org.mockito.Mockito.times
import org.mockito.Mockito.verify
import org.mockito.Mockito.never
import org.mockito.Mockito.`when`
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.BoundingBox
import software.amazon.awssdk.services.rekognition.model.DeleteCollectionRequest
import software.amazon.awssdk.services.rekognition.model.DeleteCollectionResponse
import software.amazon.awssdk.services.rekognition.model.DetectFacesRequest
import software.amazon.awssdk.services.rekognition.model.DetectFacesResponse
import software.amazon.awssdk.services.rekognition.model.DetectLabelsFeatureName
import software.amazon.awssdk.services.rekognition.model.DetectLabelsImageForeground
import software.amazon.awssdk.services.rekognition.model.DetectLabelsImageProperties
import software.amazon.awssdk.services.rekognition.model.DetectLabelsImageQuality
import software.amazon.awssdk.services.rekognition.model.DetectLabelsRequest
import software.amazon.awssdk.services.rekognition.model.DetectLabelsResponse
import software.amazon.awssdk.services.rekognition.model.InvalidImageFormatException
import software.amazon.awssdk.services.rekognition.model.Label
import software.amazon.awssdk.services.rekognition.model.DetectModerationLabelsRequest
import software.amazon.awssdk.services.rekognition.model.DetectModerationLabelsResponse
import software.amazon.awssdk.services.rekognition.model.ModerationLabel
import software.amazon.awssdk.services.rekognition.model.DetectTextRequest
import software.amazon.awssdk.services.rekognition.model.DetectTextResponse
import software.amazon.awssdk.services.rekognition.model.Face
import software.amazon.awssdk.services.rekognition.model.FaceDetail
import software.amazon.awssdk.services.rekognition.model.FaceMatch
import software.amazon.awssdk.services.rekognition.model.FaceRecord
import software.amazon.awssdk.services.rekognition.model.IndexFacesRequest
import software.amazon.awssdk.services.rekognition.model.IndexFacesResponse
import software.amazon.awssdk.services.rekognition.model.ResourceNotFoundException
import software.amazon.awssdk.services.rekognition.model.ListFacesRequest
import software.amazon.awssdk.services.rekognition.model.ListFacesResponse
import software.amazon.awssdk.services.rekognition.model.DeleteFacesRequest
import software.amazon.awssdk.services.rekognition.model.DeleteFacesResponse
import software.amazon.awssdk.services.rekognition.model.UnsuccessfulFaceDeletion
import software.amazon.awssdk.services.rekognition.model.SearchFacesByImageRequest
import software.amazon.awssdk.services.rekognition.model.SearchFacesByImageResponse
import software.amazon.awssdk.services.rekognition.model.TextDetection
import software.amazon.awssdk.services.rekognition.model.TextTypes
import software.amazon.awssdk.services.rekognition.model.ThrottlingException
import software.amazon.awssdk.services.rekognition.model.ProvisionedThroughputExceededException
import software.amazon.awssdk.services.rekognition.model.InternalServerErrorException
import software.amazon.awssdk.awscore.exception.AwsErrorDetails
import org.springframework.http.HttpStatus
import org.slf4j.LoggerFactory
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.IIOImage
import javax.imageio.ImageIO
import javax.imageio.ImageWriteParam

// Unit tests for the Rekognition → DTO mapping. No AWS network: RekognitionClient
// is mocked. Small byte arrays skip the downscaler (passthrough under 4.5 MB).
class RekognitionAiClientTest {
    private lateinit var rek: RekognitionClient
    private lateinit var client: RekognitionAiClient
    private val props = RekognitionProperties(maxImageDimension = 1024, maxCallsPerSecond = 1000.0)
    private val eventId: UUID = UUID.randomUUID()
    private val photoId: UUID = UUID.randomUUID()
    private val bytes = ByteArray(1024) { 1 }
    private lateinit var registry: SimpleMeterRegistry

    @BeforeEach
    fun setup() {
        rek = mock(RekognitionClient::class.java)
        registry = SimpleMeterRegistry()
        client = RekognitionAiClient(rek, props, registry)
    }

    // FS-2 (R3): the indexer prepares the inline image once and hands the same
    // array to enroll and bib. That is only safe if AWS receives exactly the
    // bytes each call would have built from the original itself.
    @Test
    fun `an image prepared once is sent byte for byte as each call would have built it`() {
        `when`(rek.indexFaces(any(IndexFacesRequest::class.java))).thenReturn(
            IndexFacesResponse.builder().faceRecords(FaceRecord.builder().face(Face.builder().faceId("f1").build()).build()).build(),
        )
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenReturn(DetectTextResponse.builder().build())
        val original = noisyJpeg(2400, 1800)
        assertTrue(original.size > 4_500_000, "fixture must exceed the inline-safe byte limit")

        val prepared = client.prepareInlineImage(original)
        assertTrue(prepared.size <= 4_500_000, "prepared image must pass straight through on its second wrap")
        client.facesEnroll(original, "image/jpeg", "$photoId.jpg", photoId.toString(), null, eventId)
        client.facesEnroll(prepared, "image/jpeg", "$photoId.jpg", photoId.toString(), null, eventId)
        client.bibsRecognize(original, "image/jpeg", "$photoId.jpg")
        client.bibsRecognize(prepared, "image/jpeg", "$photoId.jpg")

        val enrolls = ArgumentCaptor.forClass(IndexFacesRequest::class.java)
        verify(rek, times(2)).indexFaces(enrolls.capture())
        val texts = ArgumentCaptor.forClass(DetectTextRequest::class.java)
        verify(rek, times(2)).detectText(texts.capture())
        assertArrayEquals(enrolls.allValues[0].image().bytes().asByteArray(), enrolls.allValues[1].image().bytes().asByteArray())
        assertArrayEquals(texts.allValues[0].image().bytes().asByteArray(), texts.allValues[1].image().bytes().asByteArray())
    }

    @Test
    fun `a small image is prepared as the very same array`() {
        assertSame(bytes, client.prepareInlineImage(bytes))
    }

    // FS-2 (R5-1): a throttled or failed AWS call must say which, so late
    // photos can be traced to their cause.
    @Test
    fun `a failed AWS call is counted and logged with its error code`() {
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenThrow(
            ThrottlingException.builder()
                .awsErrorDetails(AwsErrorDetails.builder().errorCode("ThrottlingException").errorMessage("Rate exceeded").build())
                .statusCode(400).build(),
        )
        `when`(rek.detectFaces(any(DetectFacesRequest::class.java))).thenThrow(IllegalStateException("socket closed"))
        val logger = LoggerFactory.getLogger(RekognitionAiClient::class.java) as Logger
        val appender = ListAppender<ILoggingEvent>().apply { start() }
        logger.addAppender(appender)
        try {
            assertThrows<AiApiException> { client.bibsRecognize(bytes, "image/jpeg", "b.jpg") }
            assertThrows<AiApiException> { client.facesDetect(bytes, "image/jpeg", "f.jpg") }
        } finally {
            logger.detachAppender(appender)
            appender.stop()
        }
        assertEquals(1.0, registry.counter("qp.ai.error", "op", "detectText", "code", "ThrottlingException").count())
        assertEquals(1.0, registry.counter("qp.ai.error", "op", "detectFaces", "code", "IllegalStateException").count())
        val warnings = appender.list.filter { it.level == Level.WARN }.map { it.formattedMessage }
        assertTrue(warnings.any { it.contains("detectText") && it.contains("ThrottlingException") }, warnings.toString())
        assertTrue(warnings.any { it.contains("detectFaces") && it.contains("IllegalStateException") }, warnings.toString())
    }

    // UP-7: a throttled call wrote nothing, so one quick retry is safe; a
    // second throttle surfaces with the THROTTLED code the indexer reads.
    private fun throttle(): ThrottlingException = ThrottlingException.builder()
        .awsErrorDetails(AwsErrorDetails.builder().errorCode("ThrottlingException").errorMessage("Rate exceeded").build())
        .statusCode(400).build() as ThrottlingException

    private fun oneFace() = IndexFacesResponse.builder()
        .faceRecords(FaceRecord.builder().face(Face.builder().faceId("f1").build()).build()).build()

    @Test fun `a throttled IndexFaces is retried once and succeeds`() {
        `when`(rek.indexFaces(any(IndexFacesRequest::class.java))).thenThrow(throttle()).thenReturn(oneFace())
        val result = client.facesEnroll(bytes, "image/jpeg", "x.jpg", photoId.toString(), null, eventId)
        assertEquals(1, result.faces_enrolled)
        verify(rek, times(2)).indexFaces(any(IndexFacesRequest::class.java))
        assertEquals(1.0, registry.counter("qp.ai.throttle.retry", "op", "indexFaces", "code", "ThrottlingException").count())
    }

    @Test fun `a throttled DetectText is retried once`() {
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenThrow(throttle()).thenReturn(DetectTextResponse.builder().build())
        client.bibsRecognize(bytes, "image/jpeg", "b.jpg")
        verify(rek, times(2)).detectText(any(DetectTextRequest::class.java))
    }

    @Test fun `a second throttle surfaces as 503 with code THROTTLED`() {
        `when`(rek.indexFaces(any(IndexFacesRequest::class.java))).thenThrow(
            ProvisionedThroughputExceededException.builder().message("slow down").build())
        val ex = assertThrows<AiApiException> { client.facesEnroll(bytes, "image/jpeg", "x.jpg", photoId.toString(), null, eventId) }
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        assertEquals("THROTTLED", ex.aiCode)
        verify(rek, times(2)).indexFaces(any(IndexFacesRequest::class.java))
    }

    @Test fun `a failure that is not a throttle is not retried`() {
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenThrow(InternalServerErrorException.builder().message("boom").build())
        val ex = assertThrows<AiApiException> { client.bibsRecognize(bytes, "image/jpeg", "b.jpg") }
        assertEquals(null, ex.aiCode)
        verify(rek, times(1)).detectText(any(DetectTextRequest::class.java))
    }

    @Test fun `calls to one operation are spaced by the rate guard`() {
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenReturn(DetectTextResponse.builder().build())
        val paced = RekognitionAiClient(rek, props.copy(maxCallsPerSecond = 20.0), registry)
        val start = System.nanoTime()
        repeat(10) { paced.bibsRecognize(bytes, "image/jpeg", "b.jpg") }
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        assertTrue(elapsedMs >= 400, "ten calls at 20 a second must take at least 400 ms, took $elapsedMs")
    }

    @Test fun `each operation has its own bucket`() {
        `when`(rek.indexFaces(any(IndexFacesRequest::class.java))).thenReturn(oneFace())
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenReturn(DetectTextResponse.builder().build())
        val paced = RekognitionAiClient(rek, props.copy(maxCallsPerSecond = 1.0), registry)
        val start = System.nanoTime()
        paced.facesEnroll(bytes, "image/jpeg", "x.jpg", photoId.toString(), null, eventId)
        paced.bibsRecognize(bytes, "image/jpeg", "b.jpg")
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        assertTrue(elapsedMs < 500, "the first call of each operation must not wait, took $elapsedMs")
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
    fun `oversized images are sent at the configured maximum dimension`() {
        `when`(rek.indexFaces(any(IndexFacesRequest::class.java))).thenReturn(
            IndexFacesResponse.builder().faceRecords(FaceRecord.builder().face(Face.builder().faceId("f1").build()).build()).build(),
        )
        val original = noisyJpeg(2400, 1800)
        assertTrue(original.size > 4_500_000, "fixture must exceed the inline-safe byte limit")

        client.facesEnroll(original, "image/jpeg", "$photoId.jpg", photoId.toString(), null, eventId)

        val captor = ArgumentCaptor.forClass(IndexFacesRequest::class.java)
        verify(rek).indexFaces(captor.capture())
        val sent = ImageIO.read(ByteArrayInputStream(captor.value.image().bytes().asByteArray()))
        assertEquals(props.maxImageDimension, maxOf(sent.width, sent.height))
        val stripeBrightness = (0 until sent.width).map { sent.getRGB(it, 100) and 0xFF }.average()
        assertTrue(stripeBrightness > 240, "source subsampling should preserve the even-pixel white stripe phase")
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

    @Test fun `no face remains an empty public result with explicit internal outcome`() {
        `when`(rek.searchFacesByImage(any(SearchFacesByImageRequest::class.java)))
            .thenThrow(software.amazon.awssdk.services.rekognition.model.InvalidParameterException.builder().message("no face").build())
        val result = client.facesSearch(bytes, "image/jpeg", "s.jpg", eventId, 0.8, 50)
        assertTrue(result.matches.isEmpty())
        val body = com.fasterxml.jackson.module.kotlin.jacksonObjectMapper().valueToTree<com.fasterxml.jackson.databind.JsonNode>(result)
        assertEquals(true, body.path("noFaces").asBoolean())
    }

    @Test fun `missing collection is distinct from no face`() {
        `when`(rek.searchFacesByImage(any(SearchFacesByImageRequest::class.java)))
            .thenThrow(ResourceNotFoundException.builder().message("no collection").build())
        val body = com.fasterxml.jackson.module.kotlin.jacksonObjectMapper().valueToTree<com.fasterxml.jackson.databind.JsonNode>(
            client.facesSearch(bytes, "image/jpeg", "s.jpg", eventId, 0.8, 50))
        assertTrue(body.has("noFaces"), "provider outcome must distinguish empty results")
        assertEquals(false, body.path("noFaces").asBoolean())
    }

    @Test fun `saturated raw faces remain possibly truncated after external id deduplication`() {
        `when`(rek.searchFacesByImage(any(SearchFacesByImageRequest::class.java))).thenReturn(
            SearchFacesByImageResponse.builder().faceMatches(List(50) {
                FaceMatch.builder().similarity(95f).face(Face.builder().externalImageId("$eventId.$photoId").build()).build()
            }).build())
        val result = client.facesSearch(bytes, "image/jpeg", "s.jpg", eventId, 0.8, 50)
        assertEquals(1, result.matches.size)
        val body = com.fasterxml.jackson.module.kotlin.jacksonObjectMapper().valueToTree<com.fasterxml.jackson.databind.JsonNode>(result)
        assertTrue(body.path("possiblyTruncated").asBoolean())
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

    @Test
    fun `delete-by-event treats a missing collection as already deleted`() {
        `when`(rek.deleteCollection(any(DeleteCollectionRequest::class.java)))
            .thenThrow(ResourceNotFoundException.builder().message("gone").build())
        client.deleteFacesByEvent(eventId) // no throw: nothing left to delete
    }

    @Test
    fun `delete-by-event throws on an AWS error so the face-expiry sweep retries`() {
        `when`(rek.deleteCollection(any(DeleteCollectionRequest::class.java)))
            .thenThrow(ThrottlingException.builder().message("slow down").build())
        kotlin.test.assertFailsWith<AiApiException> { client.deleteFacesByEvent(eventId) }
    }

    private fun noisyJpeg(width: Int, height: Int): ByteArray {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        var state = 7
        for (y in 0 until height) for (x in 0 until width) {
            if (y < 400) {
                image.setRGB(x, y, if (x % 2 == 0) 0xFFFFFF else 0)
            } else {
                state = state * 1_103_515_245 + 12_345
                image.setRGB(x, y, state)
            }
        }
        val writer = ImageIO.getImageWritersByFormatName("jpeg").next()
        val output = ByteArrayOutputStream()
        try {
            ImageIO.createImageOutputStream(output).use { stream ->
                writer.output = stream
                writer.write(null, IIOImage(image, null, null), writer.defaultWriteParam.apply {
                    compressionMode = ImageWriteParam.MODE_EXPLICIT
                    compressionQuality = 1f
                })
            }
        } finally {
            writer.dispose()
        }
        return output.toByteArray()
    }

    @Test fun `rekognitionDeletesEveryFaceOfOnlyTheRequestedPhotoAcrossPages`() {
        val faceA1 = "face-a-1"
        val faceA2 = "face-a-2"
        val faceFromOtherPhoto = "face-b-1"
        val deleted = mutableSetOf<String>()
        `when`(rek.listFaces(any(ListFacesRequest::class.java))).thenAnswer { invocation ->
            val request = invocation.getArgument<ListFacesRequest>(0)
            assertEquals("qp-event-$eventId", request.collectionId())
            if (deleted.isNotEmpty()) ListFacesResponse.builder().faces(
                Face.builder().faceId(faceFromOtherPhoto).externalImageId("$eventId.${UUID.randomUUID()}").build()).build()
            else if (request.nextToken() == "page-2") ListFacesResponse.builder().faces(
                Face.builder().faceId(faceA2).externalImageId("$eventId.$photoId").build()).build()
            else ListFacesResponse.builder().nextToken("page-2").faces(
                Face.builder().faceId(faceA1).externalImageId("$eventId.$photoId").build(),
                Face.builder().faceId(faceFromOtherPhoto).externalImageId("$eventId.${UUID.randomUUID()}").build()).build()
        }
        `when`(rek.deleteFaces(any(DeleteFacesRequest::class.java))).thenAnswer { invocation ->
            val request = invocation.getArgument<DeleteFacesRequest>(0)
            deleted.addAll(request.faceIds())
            DeleteFacesResponse.builder().deletedFaces(request.faceIds()).build()
        }

        // The compatibility entry must perform the same selective operation.
        client.deleteFacesPerson("$eventId.$photoId")

        val requestedDeletionIds = deleted.toSet()
        assertEquals(setOf(faceA1, faceA2), requestedDeletionIds)
        assertFalse(requestedDeletionIds.contains(faceFromOtherPhoto))
        verify(rek, never()).deleteCollection(any(DeleteCollectionRequest::class.java))
    }

    @Test fun `partialDeleteIsNotSuccess`() {
        `when`(rek.listFaces(any(ListFacesRequest::class.java))).thenReturn(
            ListFacesResponse.builder().faces(Face.builder().faceId("face-a").externalImageId("$eventId.$photoId").build()).build())
        `when`(rek.deleteFaces(any(DeleteFacesRequest::class.java))).thenReturn(
            DeleteFacesResponse.builder().unsuccessfulFaceDeletions(
                UnsuccessfulFaceDeletion.builder().faceId("face-a").build()).build())

        val partialDeletionWasReportedComplete = runCatching { client.deleteFacesPerson("$eventId.$photoId") }.isSuccess

        assertFalse(partialDeletionWasReportedComplete)
        verify(rek, never()).deleteCollection(any(DeleteCollectionRequest::class.java))
    }

    @Test fun `remaining face after acknowledged deletion is not confirmed absent`() {
        `when`(rek.listFaces(any(ListFacesRequest::class.java))).thenReturn(
            ListFacesResponse.builder().faces(Face.builder().faceId("face-a").externalImageId("$eventId.$photoId").build()).build())
        `when`(rek.deleteFaces(any(DeleteFacesRequest::class.java))).thenReturn(
            DeleteFacesResponse.builder().deletedFaces("face-a").build())
        assertFalse(runCatching { client.deleteFacesPerson("$eventId.$photoId") }.isSuccess)
    }

    @Test fun `malformed composite target cannot silently succeed`() {
        assertFalse(runCatching { client.deleteFacesPerson("not-an-event.not-a-photo") }.isSuccess)
    }

    // AA-I11: profile-image moderation reuses the pacing, retry and error mapping.
    @Test fun `moderationLabels sends MinConfidence on the 0-100 scale and maps names and parents`() {
        `when`(rek.detectModerationLabels(any(DetectModerationLabelsRequest::class.java))).thenReturn(
            DetectModerationLabelsResponse.builder().moderationLabels(
                ModerationLabel.builder().name("Explicit").parentName("").confidence(97.5f).build(),
                ModerationLabel.builder().name("Exposed Male Genitalia").parentName("Explicit").confidence(95f).build(),
            ).build(),
        )

        val hits = client.moderationLabels(bytes, 0.8)

        val req = ArgumentCaptor.forClass(DetectModerationLabelsRequest::class.java)
        verify(rek).detectModerationLabels(req.capture())
        assertEquals(80f, req.value.minConfidence())
        assertArrayEquals(bytes, req.value.image().bytes().asByteArray())
        assertEquals(
            listOf(ModerationHit("Explicit", null, 0.975), ModerationHit("Exposed Male Genitalia", "Explicit", 0.95)),
            hits,
        )
    }

    @Test fun `moderationLabels retries once when throttled`() {
        `when`(rek.detectModerationLabels(any(DetectModerationLabelsRequest::class.java)))
            .thenThrow(throttle()).thenReturn(DetectModerationLabelsResponse.builder().build())

        assertEquals(emptyList<ModerationHit>(), client.moderationLabels(bytes, 0.8))
        verify(rek, times(2)).detectModerationLabels(any(DetectModerationLabelsRequest::class.java))
    }

    // RV-1.3: the Photo Relevance v2 image check (DetectLabels with image properties).
    private fun quality(sharpness: Float, brightness: Float) =
        DetectLabelsImageQuality.builder().sharpness(sharpness).brightness(brightness).build()

    private fun labelsResponse(withForeground: Boolean = true) = DetectLabelsResponse.builder()
        .labels(
            Label.builder().name("Person").confidence(98.5f).build(),
            Label.builder().name("Marathon").confidence(72f).build(),
        )
        .imageProperties(
            DetectLabelsImageProperties.builder()
                .quality(quality(41.5f, 63f))
                .apply { if (withForeground) foreground(DetectLabelsImageForeground.builder().quality(quality(88f, 55.25f)).build()) }
                .build(),
        )
        .build()

    private fun smallJpeg(width: Int, height: Int): ByteArray {
        val out = ByteArrayOutputStream()
        ImageIO.write(BufferedImage(width, height, BufferedImage.TYPE_INT_RGB), "jpeg", out)
        return out.toByteArray()
    }

    @Test fun `the image check asks for labels and image properties`() {
        `when`(rek.detectLabels(any(DetectLabelsRequest::class.java))).thenReturn(labelsResponse())
        client.imageCheck(smallJpeg(200, 100))
        val req = ArgumentCaptor.forClass(DetectLabelsRequest::class.java)
        verify(rek).detectLabels(req.capture())
        assertEquals(setOf(DetectLabelsFeatureName.GENERAL_LABELS, DetectLabelsFeatureName.IMAGE_PROPERTIES), req.value.features().toSet())
    }

    @Test fun `it maps the foreground and whole-image scores and the labels`() {
        `when`(rek.detectLabels(any(DetectLabelsRequest::class.java))).thenReturn(labelsResponse())
        val check = client.imageCheck(smallJpeg(200, 100))
        assertEquals(
            ImageCheck(
                fgSharpness = 88.0, fgBrightness = 55.25, sharpness = 41.5, brightness = 63.0,
                labels = listOf(LabelHit("Person", 0.985), LabelHit("Marathon", 0.72)),
            ),
            check,
        )
    }

    @Test fun `a photo with no foreground gives null foreground scores`() {
        `when`(rek.detectLabels(any(DetectLabelsRequest::class.java))).thenReturn(labelsResponse(withForeground = false))
        val check = client.imageCheck(smallJpeg(200, 100))
        assertEquals(null, check.fgSharpness)
        assertEquals(null, check.fgBrightness)
        assertEquals(41.5, check.sharpness)
        assertEquals(63.0, check.brightness)
    }

    @Test fun `a small file is still shrunk to the fixed size`() {
        `when`(rek.detectLabels(any(DetectLabelsRequest::class.java))).thenReturn(labelsResponse())
        val original = smallJpeg(2000, 1500)
        assertTrue(original.size < 4_500_000, "fixture must be under the inline-safe byte limit")
        client.imageCheck(original)
        val req = ArgumentCaptor.forClass(DetectLabelsRequest::class.java)
        verify(rek).detectLabels(req.capture())
        val sent = ImageIO.read(ByteArrayInputStream(req.value.image().bytes().asByteArray()))
        assertEquals(props.maxImageDimension, maxOf(sent.width, sent.height))
    }

    @Test fun `a throttle is retried once`() {
        `when`(rek.detectLabels(any(DetectLabelsRequest::class.java))).thenThrow(throttle()).thenReturn(labelsResponse())
        client.imageCheck(smallJpeg(200, 100))
        verify(rek, times(2)).detectLabels(any(DetectLabelsRequest::class.java))
        assertEquals(1.0, registry.counter("qp.ai.throttle.retry", "op", "detectLabels", "code", "ThrottlingException").count())
    }

    @Test fun `an image AWS rejects becomes a 422`() {
        `when`(rek.detectLabels(any(DetectLabelsRequest::class.java)))
            .thenThrow(InvalidImageFormatException.builder().message("bad image").build())
        val ex = assertThrows<AiApiException> { client.imageCheck(smallJpeg(200, 100)) }
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, ex.status)
    }

    @Test fun `the image check waits on its own limiter`() {
        `when`(rek.detectText(any(DetectTextRequest::class.java))).thenReturn(DetectTextResponse.builder().build())
        `when`(rek.detectLabels(any(DetectLabelsRequest::class.java))).thenReturn(labelsResponse())
        val paced = RekognitionAiClient(rek, props.copy(maxCallsPerSecond = 1.0), registry)
        val image = smallJpeg(200, 100)
        val start = System.nanoTime()
        paced.bibsRecognize(bytes, "image/jpeg", "b.jpg")
        paced.imageCheck(image)
        assertTrue((System.nanoTime() - start) / 1_000_000 < 500, "the first image check must not wait on the text limiter")
        val second = System.nanoTime()
        paced.imageCheck(image)
        assertTrue((System.nanoTime() - second) / 1_000_000 >= 500, "a second image check in the same second must wait for its own bucket")
    }
}
