package com.quickpitik.service.ai

import com.quickpitik.config.IdentityProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.stereotype.Service
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.CreateFaceLivenessSessionRequest
import software.amazon.awssdk.services.rekognition.model.CreateFaceLivenessSessionRequestSettings
import software.amazon.awssdk.services.rekognition.model.GetFaceLivenessSessionResultsRequest
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import software.amazon.awssdk.services.sts.StsClient
import software.amazon.awssdk.services.sts.model.AssumeRoleRequest
import software.amazon.awssdk.services.sts.model.Credentials

/** One finished (or not) live check. [referenceImage] is JPEG bytes; nothing is stored here. */
class LivenessResult(val status: LivenessSessionStatus, val confidence: Float?, val referenceImage: ByteArray?)

// AWS Face Liveness for the identity checks (AA-I2). No face collection is used and
// nothing is indexed. Callers keep every call outside a DB transaction and map SdkException.
@Service
@ConditionalOnProperty(prefix = "app.ai", name = ["provider"], havingValue = "rekognition")
class RekognitionIdentityClient(
    @Qualifier("livenessRekognitionClient") private val liveness: RekognitionClient,
    private val sts: StsClient,
    private val props: IdentityProperties,
) {
    /** No audit images and no S3 output (owner answer 7): AWS keeps nothing for us. */
    fun createLivenessSession(): String = liveness.createFaceLivenessSession(
        CreateFaceLivenessSessionRequest.builder()
            .settings(CreateFaceLivenessSessionRequestSettings.builder().auditImagesLimit(0).build())
            .build(),
    ).sessionId()

    fun livenessResult(sessionId: String): LivenessResult {
        val r = liveness.getFaceLivenessSessionResults(GetFaceLivenessSessionResultsRequest.builder().sessionId(sessionId).build())
        return LivenessResult(r.status(), r.confidence(), r.referenceImage()?.bytes()?.asByteArray())
    }

    /** Short-lived keys that can only start a liveness check in [IdentityProperties.livenessRegion] (AA-0.2). */
    fun browserCredentials(sessionId: String): Credentials = sts.assumeRole(
        AssumeRoleRequest.builder()
            .roleArn(props.livenessRoleArn)
            // Shows up in CloudTrail: the AWS session id only, never a name or email.
            .roleSessionName("lv-$sessionId".take(64))
            .durationSeconds(props.livenessCredentialSeconds)
            .policy(startOnlyPolicy())
            .build(),
    ).credentials()

    private fun startOnlyPolicy() = """{"Version":"2012-10-17","Statement":[{"Effect":"Allow",""" +
        """"Action":"rekognition:StartFaceLivenessSession","Resource":"*",""" +
        """"Condition":{"StringEquals":{"aws:RequestedRegion":"${props.livenessRegion}"}}}]}"""
}
