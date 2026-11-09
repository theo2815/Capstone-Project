package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerReasonCode
import com.quickpitik.service.ai.RekognitionIdentityClient
import com.quickpitik.service.liveness.LivenessSessionService
import org.springframework.beans.factory.ObjectProvider
import org.springframework.stereotype.Service
import software.amazon.awssdk.core.exception.SdkException
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.OffsetDateTime
import java.util.UUID

/** What the automatic identity checks found. [reasons] is empty when every check passed; scores are 0-100. */
class IdentityChecks(
    val reasons: List<SellerReasonCode>,
    val livenessConfidence: BigDecimal? = null,
    val faceSimilarity: BigDecimal? = null,
    /** The live check's reference still, to be stored privately with the ID photos (owner answer 7). */
    val referenceImage: ByteArray? = null,
    val sessionId: String? = null,
    /** When the consent tick was given: the session's creation time. */
    val consentAt: OffsetDateTime? = null,
)

/**
 * The automatic identity decision (AA-I3): the liveness score, then CompareFaces with the live
 * reference as source and the ID front as target. Never the holding-ID photo. Checks only pass
 * quietly or send the seller to an admin; only an admin rejects. Call with no transaction open.
 */
@Service
class IdentityCheckService(
    private val livenessSessionService: LivenessSessionService,
    private val identityClient: ObjectProvider<RekognitionIdentityClient>,
    private val aiApiProperties: AiApiProperties,
    private val identityProperties: IdentityProperties,
) {
    /** [sessionId] null = "Have an admin check instead". [idFront] is read only if the compare runs. */
    fun run(userId: UUID, sessionId: String?, idFront: () -> ByteArray): IdentityChecks {
        if (sessionId == null) return IdentityChecks(listOf(SellerReasonCode.DECLINED))
        val session = livenessSessionService.requireOpen(userId, sessionId, PURPOSE, null)
        // Claim before reading (AA-I2's order): a parallel submit with the same session stops here, unbilled.
        if (!livenessSessionService.claim(userId, sessionId, PURPOSE, null)) throw LivenessSessionService.invalidSession()
        fun checks(reasons: List<SellerReasonCode>, confidence: Float? = null, similarity: Float? = null, image: ByteArray? = null) =
            IdentityChecks(reasons, confidence?.score(), similarity?.score(), image, sessionId, session.createdAt)

        val client = identityClient.getIfAvailable()
        if (!aiApiProperties.enabled || client == null) return checks(listOf(SellerReasonCode.AI_DISABLED))
        val result = try {
            client.livenessResult(sessionId)
        } catch (ex: SdkException) {
            return checks(listOf(SellerReasonCode.AI_ERROR))
        }
        // Unfinished, failed or expired: nothing to decide on. The session stays used; start a new one.
        if (result.status != LivenessSessionStatus.SUCCEEDED) throw LivenessSessionService.invalidSession()

        val reasons = mutableListOf<SellerReasonCode>()
        val confidence = result.confidence
        if (confidence == null || confidence < identityProperties.livenessMinConfidence) reasons += SellerReasonCode.LIVENESS_LOW
        // Compare runs even after LIVENESS_LOW, so the admin sees both scores. A SUCCEEDED
        // session with no reference image is an AWS fault like any other (AI_ERROR).
        val compare = try {
            result.referenceImage?.let { client.compareFaces(it, idFront()) }
        } catch (ex: SdkException) {
            null
        }
        val similarity = compare?.bestSimilarity
        when {
            compare == null -> reasons += SellerReasonCode.AI_ERROR
            compare.facesOnTarget == 0 -> reasons += SellerReasonCode.NO_FACE_ON_ID
            similarity == null || similarity < identityProperties.faceMinSimilarity -> reasons += SellerReasonCode.FACE_MISMATCH
        }
        return checks(reasons, confidence, similarity, result.referenceImage)
    }

    companion object {
        private val PURPOSE = LivenessPurpose.PHOTOGRAPHER_IDENTITY

        /** No failing reason = AUTO_APPROVED; anything else waits for an admin. */
        fun statusFor(reasons: List<SellerReasonCode>): SellerDetailsStatus =
            if (reasons.all { it == SellerReasonCode.PASSED }) SellerDetailsStatus.AUTO_APPROVED else SellerDetailsStatus.MANUAL_REVIEW

        private fun Float.score(): BigDecimal = toBigDecimal().setScale(2, RoundingMode.HALF_UP)
    }
}
