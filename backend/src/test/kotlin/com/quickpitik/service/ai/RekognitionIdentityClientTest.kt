package com.quickpitik.service.ai

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.IdentityProperties
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito.mock
import org.mockito.Mockito.`when`
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.AuditImage
import software.amazon.awssdk.services.rekognition.model.CreateFaceLivenessSessionRequest
import software.amazon.awssdk.services.rekognition.model.CreateFaceLivenessSessionResponse
import software.amazon.awssdk.services.rekognition.model.GetFaceLivenessSessionResultsRequest
import software.amazon.awssdk.services.rekognition.model.GetFaceLivenessSessionResultsResponse
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import software.amazon.awssdk.services.sts.StsClient
import software.amazon.awssdk.services.sts.model.AssumeRoleRequest
import software.amazon.awssdk.services.sts.model.AssumeRoleResponse
import software.amazon.awssdk.services.sts.model.Credentials
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNull

class RekognitionIdentityClientTest {
    private val rekognition = mock(RekognitionClient::class.java)
    private val sts = mock(StsClient::class.java)
    private val props = IdentityProperties(livenessRoleArn = "arn:aws:iam::000000000000:role/quickpitik-liveness-browser")
    private val client = RekognitionIdentityClient(rekognition, sts, props)

    @Test
    fun `the liveness session asks for no audit images and no output bucket`() {
        val captor = ArgumentCaptor.forClass(CreateFaceLivenessSessionRequest::class.java)
        `when`(rekognition.createFaceLivenessSession(captor.capture()))
            .thenReturn(CreateFaceLivenessSessionResponse.builder().sessionId("s-1").build())

        assertEquals("s-1", client.createLivenessSession())

        val settings = captor.value.settings()
        assertEquals(0, settings.auditImagesLimit())
        assertNull(settings.outputConfig(), "owner answer 7: AWS stores nothing in S3 for us")
    }

    @Test
    fun `the result returns the reference image bytes and the score`() {
        val jpeg = byteArrayOf(1, 2, 3)
        `when`(rekognition.getFaceLivenessSessionResults(any(GetFaceLivenessSessionResultsRequest::class.java))).thenReturn(
            GetFaceLivenessSessionResultsResponse.builder().sessionId("s-1").status(LivenessSessionStatus.SUCCEEDED).confidence(77.1f)
                .referenceImage(AuditImage.builder().bytes(SdkBytes.fromByteArray(jpeg)).build()).build(),
        )

        val result = client.livenessResult("s-1")

        assertEquals(LivenessSessionStatus.SUCCEEDED, result.status)
        assertEquals(77.1f, result.confidence)
        assertContentEquals(jpeg, result.referenceImage)
    }

    @Test
    fun `browser keys are a 900 second role session that may only start a liveness check in ap-southeast-5`() {
        val captor = ArgumentCaptor.forClass(AssumeRoleRequest::class.java)
        val keys = Credentials.builder().accessKeyId("a").secretAccessKey("b").sessionToken("c").build()
        `when`(sts.assumeRole(captor.capture())).thenReturn(AssumeRoleResponse.builder().credentials(keys).build())

        assertEquals(keys, client.browserCredentials("4f6a1c0e-0000-4000-8000-000000000001"))

        val req = captor.value
        assertEquals(props.livenessRoleArn, req.roleArn())
        assertEquals(900, req.durationSeconds())
        assertEquals("lv-4f6a1c0e-0000-4000-8000-000000000001", req.roleSessionName())
        val statement = ObjectMapper().readTree(req.policy()).path("Statement").single()
        assertEquals("rekognition:StartFaceLivenessSession", statement.path("Action").asText())
        assertEquals("ap-southeast-5", statement.path("Condition").path("StringEquals").path("aws:RequestedRegion").asText())
    }
}
