package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerIdType
import com.quickpitik.entity.SellerReasonCode
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.ai.RekognitionIdentityClient
import com.quickpitik.service.liveness.LivenessSessionService
import org.springframework.beans.factory.ObjectProvider
import org.springframework.stereotype.Service
import software.amazon.awssdk.core.exception.SdkException
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import java.math.BigDecimal
import java.math.RoundingMode
import java.nio.charset.StandardCharsets
import java.time.OffsetDateTime
import java.util.HexFormat
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

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
    /** The keyed hash of the typed ID number (AA-I4); null when none was typed. The raw number is never kept. */
    val idNumberHmac: String? = null,
    /** The typed number's hash under every [SellerIdType], in entries order, for the repeat match (R-1); empty when none was typed. */
    val idNumberMatchHmacs: List<String> = emptyList(),
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
    /**
     * [sessionId] null = "Have an admin check instead": no OCR, but a typed [idNumber] is still hashed (AA-I5).
     * A malformed [idNumber] is refused before the session is used. [idFront] is read at most once.
     */
    fun run(userId: UUID, sessionId: String?, idType: SellerIdType, idNumber: String?, idFront: () -> ByteArray): IdentityChecks {
        val number = idNumber?.takeIf { it.isNotBlank() }?.let { normaliseIdNumber(idType, it) }
        val hmac = number?.let { idNumberHmac(idType, it) }
        val matchHmacs = number?.let { n -> SellerIdType.entries.map { idNumberHmac(it, n) } }.orEmpty()
        if (sessionId == null) return IdentityChecks(listOf(SellerReasonCode.DECLINED), idNumberHmac = hmac, idNumberMatchHmacs = matchHmacs)
        val session = livenessSessionService.requireOpen(userId, sessionId, PURPOSE, null)
        // Claim before reading (AA-I2's order): a parallel submit with the same session stops here, unbilled.
        if (!livenessSessionService.claim(userId, sessionId, PURPOSE, null)) throw LivenessSessionService.invalidSession()
        fun checks(reasons: List<SellerReasonCode>, confidence: Float? = null, similarity: Float? = null, image: ByteArray? = null) =
            IdentityChecks(reasons, confidence?.score(), similarity?.score(), image, sessionId, session.createdAt, hmac, matchHmacs)
        val front by lazy(idFront)

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
            result.referenceImage?.let { client.compareFaces(it, front) }
        } catch (ex: SdkException) {
            null
        }
        val similarity = compare?.bestSimilarity
        when {
            compare == null -> reasons += SellerReasonCode.AI_ERROR
            compare.facesOnTarget == 0 -> reasons += SellerReasonCode.NO_FACE_ON_ID
            similarity == null || similarity < identityProperties.faceMinSimilarity -> reasons += SellerReasonCode.FACE_MISMATCH
        }
        idNumberReason(client, number) { front }?.takeUnless { it in reasons }?.let { reasons += it }
        // No format for this type: a card word ("REPUBLIC") would pass the line match, so an admin checks (R-1).
        if (number != null && identityProperties.idNumberPatterns[idType] == null) reasons += SellerReasonCode.ID_FORMAT_UNCHECKED
        return checks(reasons, confidence, similarity, result.referenceImage)
    }

    /** Uppercase, no spaces or dashes, 4-30 letters/digits, then the ID type's format when one is set. */
    fun normaliseIdNumber(idType: SellerIdType, typed: String): String {
        val number = typed.uppercase().replace(SPACES_AND_DASHES, "")
        val format = identityProperties.idNumberPatterns[idType]
        // The message never repeats the number.
        if (!ID_NUMBER_SHAPE.matches(number) || (format != null && !format.matches(number))) {
            throw ValidationException("Enter the ID number exactly as it is printed on your ID.", field = "idNumber")
        }
        return number
    }

    /** HMAC-SHA256(ID_NUMBER_HMAC_SECRET, "<ID_TYPE>:<normalised>") as lowercase hex: the only form ever stored. */
    fun idNumberHmac(idType: SellerIdType, normalised: String): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(identityProperties.idNumberHmacSecret.toByteArray(StandardCharsets.UTF_8), "HmacSHA256"))
        return HexFormat.of().formatHex(mac.doFinal("${idType.name}:$normalised".toByteArray(StandardCharsets.UTF_8)))
    }

    /** The number must sit whole in one line of the ID front: never a part of it, never pieced across lines (review F8). */
    private fun idNumberReason(client: RekognitionIdentityClient, number: String?, front: () -> ByteArray): SellerReasonCode? {
        if (number == null) return SellerReasonCode.ID_NUMBER_UNREADABLE
        val lines = try {
            client.detectLines(front())
        } catch (ex: SdkException) {
            return SellerReasonCode.AI_ERROR
        }
        return when {
            lines.isEmpty() -> SellerReasonCode.ID_NUMBER_UNREADABLE
            lines.none { lineHolds(it, number) } -> SellerReasonCode.ID_NUMBER_MISMATCH
            else -> null
        }
    }

    companion object {
        private val SPACES_AND_DASHES = Regex("[\\s-]")
        private val ID_NUMBER_SHAPE = Regex("[A-Z0-9]{4,30}")

        /** True when [number] equals one token, or a run of consecutive tokens joined, of this one line. */
        private fun lineHolds(line: String, number: String): Boolean {
            val tokens = line.split(Regex("\\s+")).map { it.uppercase().replace("-", "") }.filter { it.isNotEmpty() }
            return tokens.indices.any { start ->
                val joined = StringBuilder()
                tokens.drop(start).any { joined.append(it).contentEquals(number) }
            }
        }

        private val PURPOSE = LivenessPurpose.PHOTOGRAPHER_IDENTITY

        /** No failing reason = AUTO_APPROVED; anything else waits for an admin. */
        fun statusFor(reasons: List<SellerReasonCode>): SellerDetailsStatus =
            if (reasons.all { it == SellerReasonCode.PASSED }) SellerDetailsStatus.AUTO_APPROVED else SellerDetailsStatus.MANUAL_REVIEW

        private fun Float.score(): BigDecimal = toBigDecimal().setScale(2, RoundingMode.HALF_UP)
    }
}
