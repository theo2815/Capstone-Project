package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.profile.SelfieRefDto
import com.quickpitik.entity.UserSelfie
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.UUID

/**
 * Selfie library — runner uploads sharp, well-lit selfies that the photo
 * grid uses to match against enrolled face embeddings (Phase B-Photos).
 *
 * Quality gate (Q-006 RESOLVED 2026-05-09): every upload calls ai-api
 * `faces/detect`. The 422 SELFIE_REJECTED carries a human-readable reason —
 * verbatim from ai-api when it had a strong opinion ("LOW_QUALITY",
 * "NO_FACES"), otherwise inferred from the detect result (zero faces,
 * multiple faces, low confidence).
 *
 * Cap=5 per Q-006 — clients lie, so the count comes from the DB. First
 * selfie auto-promotes to primary; deleting the primary auto-promotes the
 * most-recently-uploaded remaining selfie. Set-primary returns the full
 * canonical list so the FE can revert atomically on failure.
 */
@Service
@Transactional
class SelfieService(
    private val userSelfieRepository: UserSelfieRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun list(userId: UUID): List<SelfieRefDto> =
        userSelfieRepository.findByUserIdOrderByUploadedAtDesc(userId).map { it.toDto() }

    fun upload(userId: UUID, file: ByteArray, contentType: String?, filename: String?): SelfieRefDto {
        if (file.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "file is empty",
                field = "file",
            )
        }
        val mime = (contentType ?: "").lowercase()
        if (mime !in SUPPORTED_TYPES) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "selfie must be jpeg, png, or webp",
                field = "file",
            )
        }
        val existingCount = userSelfieRepository.countByUserId(userId)
        if (existingCount >= MAX_SELFIES) {
            throw ConflictException(
                code = ErrorCodes.SELFIE_LIMIT_REACHED,
                message = "Selfie limit of $MAX_SELFIES reached — delete one before uploading another.",
            )
        }

        val qualityScore = qualityGate(file, mime, filename ?: "selfie")

        val selfieId = UUID.randomUUID()
        val key = "selfies/$userId/$selfieId.${extensionOf(mime)}"
        storageService.put(key, file, mime)

        val isFirst = existingCount == 0L
        val saved = userSelfieRepository.save(
            UserSelfie(
                id = selfieId,
                userId = userId,
                s3Key = key,
                isPrimary = isFirst,
                qualityScore = qualityScore,
            ),
        )
        return saved.toDto()
    }

    /**
     * DELETE is idempotent — return false (no row removed) instead of 404 when missing.
     *
     * M-2 — There is a 1-statement gap between the delete+flush and the
     * subsequent promote-most-recent path. Concurrency is bounded by the
     * partial unique index `uq_user_selfies_primary_per_user (user_id) WHERE
     * is_primary = true` — Postgres enforces at-most-one primary per user.
     * Two concurrent DELETEs on the same user serialize via this transactional
     * method (Spring + JPA per-call); a concurrent INSERT with isPrimary=true
     * either commits before our promote (in which case our promote becomes a
     * no-op when the new row's flush already won) or collides on the unique
     * index and surfaces a DataIntegrityViolation. In either case we never
     * leave the row set with two primaries — the worst case is a brief
     * "zero primaries" window which is also the explicit terminal state when
     * the runner deletes their last selfie. No SELECT … FOR UPDATE needed.
     */
    fun delete(userId: UUID, selfieId: UUID): Boolean {
        val selfie = userSelfieRepository.findByIdAndUserId(selfieId, userId) ?: return false
        val wasPrimary = selfie.isPrimary
        userSelfieRepository.delete(selfie)
        userSelfieRepository.flush()
        runCatching { storageService.delete(selfie.s3Key) }
            .onFailure { log.warn("selfie delete of {} failed: {}", selfie.s3Key, it.message) }
        if (wasPrimary) {
            userSelfieRepository.findFirstByUserIdOrderByUploadedAtDesc(userId)?.let { promoted ->
                promoted.isPrimary = true
                userSelfieRepository.save(promoted)
            }
        }
        return true
    }

    fun setPrimary(userId: UUID, selfieId: UUID): List<SelfieRefDto> {
        val target = userSelfieRepository.findByIdAndUserId(selfieId, userId)
            ?: throw NotFoundException(
                code = ErrorCodes.SELFIE_NOT_FOUND,
                message = "Selfie not found",
            )
        if (!target.isPrimary) {
            // Demote first to satisfy the partial unique index, then promote.
            userSelfieRepository.demoteOtherPrimaries(userId, target.id)
            userSelfieRepository.flush()
            target.isPrimary = true
            userSelfieRepository.save(target)
        }
        return list(userId)
    }

    private fun qualityGate(file: ByteArray, contentType: String, filename: String): BigDecimal {
        if (!aiApiProperties.enabled) {
            // Feature-dev short-circuit. Selfie still uploads; quality score is a
            // placeholder until ai-api comes online. Existing selfies are NOT
            // back-filled when AI is enabled later — they keep qualityScore=0
            // until the user re-uploads.
            log.debug("ai-api disabled; skipping selfie quality gate")
            return BigDecimal.ZERO
        }
        val result = try {
            aiApiClient.facesDetect(file = file, contentType = contentType, filename = filename)
        } catch (ex: AiApiException) {
            log.warn("ai-api faces/detect failed during selfie upload: code={} msg={}", ex.aiCode, ex.message)
            if (ex.aiCode == "LOW_QUALITY" || ex.aiCode == "NO_FACES") {
                throw ValidationException(
                    code = ErrorCodes.SELFIE_REJECTED,
                    message = ex.message ?: "Selfie rejected",
                    field = "file",
                )
            }
            throw ex
        }

        if (result.faces.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.SELFIE_REJECTED,
                message = "No face detected — make sure your face is centered and well-lit.",
                field = "file",
            )
        }
        if (result.faces.size > 1) {
            throw ValidationException(
                code = ErrorCodes.SELFIE_REJECTED,
                message = "Multiple faces detected — selfies must show only your face.",
                field = "file",
            )
        }
        val confidence = BigDecimal.valueOf(result.faces.first().confidence)
            .setScale(4, RoundingMode.HALF_UP)
        if (confidence < MIN_QUALITY_SCORE) {
            throw ValidationException(
                code = ErrorCodes.SELFIE_REJECTED,
                message = "Image quality too low — try a sharper, better-lit selfie.",
                field = "file",
            )
        }
        return confidence
    }

    private fun UserSelfie.toDto(): SelfieRefDto = SelfieRefDto(
        id = id,
        dataUrl = storageService.presignedGetUrl(s3Key, storageProperties.presignedTtl.selfie),
        uploadedAt = uploadedAt,
        isPrimary = isPrimary,
        qualityScore = qualityScore,
    )

    private fun extensionOf(mime: String): String = when (mime) {
        "image/jpeg", "image/jpg" -> "jpg"
        "image/png" -> "png"
        "image/webp" -> "webp"
        else -> "bin"
    }

    private companion object {
        const val MAX_SELFIES = 5
        val MIN_QUALITY_SCORE: BigDecimal = BigDecimal("0.6000")
        val SUPPORTED_TYPES = setOf("image/jpeg", "image/jpg", "image/png", "image/webp")
    }
}
