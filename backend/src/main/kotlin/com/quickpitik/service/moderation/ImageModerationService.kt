package com.quickpitik.service.moderation

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.ModerationProperties
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import org.springframework.beans.factory.ObjectProvider
import org.springframework.stereotype.Service
import java.time.Instant
import java.util.UUID

enum class ImageModerationVerdict { CLEAN, FLAGGED, UNCHECKED }

data class ImageModerationResult(val verdict: ImageModerationVerdict, val labels: List<String> = emptyList())

// AA-I11: checks a profile image before it is stored. UNCHECKED (AI off, no
// Rekognition provider, an AWS error) never refuses; the caller flags it for an admin.
// Contract: never call check inside a DB transaction; pass the processed bytes you will store.
@Service
class ImageModerationService(
    private val aiApiProperties: AiApiProperties,
    private val props: ModerationProperties,
    private val rekognition: ObjectProvider<RekognitionAiClient>,
    private val rateLimiter: RateLimiter,
) {
    private val blocked = props.imageBlockedLabels.map { it.trim().lowercase() }.filter { it.isNotEmpty() }.toSet()

    // R-14: the billed check, capped per user per day. Over the cap: UNCHECKED, no AWS call (the caller flags it).
    // shortcut: in-memory daily cap, resets on deploy; move to a stored count if abuse is seen
    fun checkFor(userId: UUID, imageBytes: ByteArray): ImageModerationResult {
        if (!rateLimiter.tryAcquire(Bucket4jRateLimiter.POLICY_IMAGE_MODERATION_DAILY, userId.toString()).allowed) return UNCHECKED
        return check(imageBytes)
    }

    fun check(imageBytes: ByteArray): ImageModerationResult {
        if (!aiApiProperties.enabled) return UNCHECKED
        val client = rekognition.getIfAvailable() ?: return UNCHECKED
        val hits = try {
            client.moderationLabels(imageBytes, props.imageMinConfidence)
        } catch (ex: AiApiException) {
            return UNCHECKED // already counted as qp.ai.error{op=detectModerationLabels}
        }
        val labels = hits
            .filter { it.name.lowercase() in blocked || it.parentName?.lowercase() in blocked }
            .map { it.name }
            .distinct()
        return if (labels.isEmpty()) CLEAN else ImageModerationResult(ImageModerationVerdict.FLAGGED, labels)
    }

    private companion object {
        val CLEAN = ImageModerationResult(ImageModerationVerdict.CLEAN)
        val UNCHECKED = ImageModerationResult(ImageModerationVerdict.UNCHECKED)
    }
}

// The one rule for the three profile-image uploads (avatar, cover, watermark).
object ProfileImageModeration {
    /**
     * Call with NO transaction open. FLAGGED: raises PROFILE_IMAGE_FLAGGED (or adds a
     * timestamped line to the open one) and throws IMAGE_NOT_ALLOWED; the flag commits
     * on its own. Otherwise returns the verdict for [flagIfUnchecked] inside the save.
     */
    fun checkOrRefuse(
        moderation: ImageModerationService,
        flags: SystemFlagService,
        userId: UUID,
        what: String,
        bytes: ByteArray,
    ): ImageModerationVerdict {
        val result = moderation.checkFor(userId, bytes)
        if (result.verdict != ImageModerationVerdict.FLAGGED) return result.verdict
        val note = "$what: ${result.labels.joinToString(", ")}"
        if (!flags.raise(FlagTargetKind.USER, userId, SystemFlagReason.PROFILE_IMAGE_FLAGGED, note)) {
            flags.appendNote(FlagTargetKind.USER, userId, SystemFlagReason.PROFILE_IMAGE_FLAGGED, "${Instant.now()} $note")
        }
        throw ValidationException(
            code = ErrorCodes.IMAGE_NOT_ALLOWED,
            message = "This image can't be used. Please choose another.",
            field = "file",
        )
    }

    /** Inside the save's transaction, so the flag commits with the stored image key. */
    fun flagIfUnchecked(flags: SystemFlagService, verdict: ImageModerationVerdict?, userId: UUID, what: String) {
        if (verdict == ImageModerationVerdict.UNCHECKED) {
            flags.raise(FlagTargetKind.USER, userId, SystemFlagReason.PROFILE_IMAGE_UNCHECKED, what)
        }
    }
}
