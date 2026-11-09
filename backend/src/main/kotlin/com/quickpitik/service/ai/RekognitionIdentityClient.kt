package com.quickpitik.service.ai

import com.quickpitik.config.IdentityProperties
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.stereotype.Service
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.CompareFacesRequest
import software.amazon.awssdk.services.rekognition.model.CreateFaceLivenessSessionRequest
import software.amazon.awssdk.services.rekognition.model.Image
import software.amazon.awssdk.services.rekognition.model.CreateFaceLivenessSessionRequestSettings
import software.amazon.awssdk.services.rekognition.model.GetFaceLivenessSessionResultsRequest
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import software.amazon.awssdk.services.sts.StsClient
import software.amazon.awssdk.services.sts.model.AssumeRoleRequest
import software.amazon.awssdk.services.sts.model.Credentials

/** One finished (or not) live check. [referenceImage] is JPEG bytes; nothing is stored here. */
class LivenessResult(val status: LivenessSessionStatus, val confidence: Float?, val referenceImage: ByteArray?)

/** Faces found on the target, and the best similarity (0-100) among them; null when there is none. */
data class FaceCompareResult(val facesOnTarget: Int, val bestSimilarity: Float?)

// AWS Face Liveness and CompareFaces for the identity checks (AA-I2, AA-I3). No face collection
// is used and nothing is indexed. Callers keep every call outside a DB transaction and map SdkException.
@Service
@ConditionalOnProperty(prefix = "app.ai", name = ["provider"], havingValue = "rekognition")
class RekognitionIdentityClient(
    @Qualifier("livenessRekognitionClient") private val liveness: RekognitionClient,
    // The @Primary client in app.ai.rekognition.region (ap-southeast-1).
    private val rekognition: RekognitionClient,
    private val sts: StsClient,
    private val props: IdentityProperties,
) {
    /**
     * Threshold 0, so every face on the target comes back with a score; the best one counts
     * (many PH IDs carry a small second "ghost" photo).
     */
    fun compareFaces(source: ByteArray, target: ByteArray): FaceCompareResult {
        val r = rekognition.compareFaces(
            CompareFacesRequest.builder()
                .sourceImage(Image.builder().bytes(SdkBytes.fromByteArray(source)).build())
                .targetImage(Image.builder().bytes(SdkBytes.fromByteArray(target)).build())
                .similarityThreshold(0f)
                .build(),
        )
        return FaceCompareResult(r.faceMatches().size + r.unmatchedFaces().size, r.faceMatches().maxOfOrNull { it.similarity() })
    }

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
