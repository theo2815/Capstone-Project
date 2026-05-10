package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.BrandPatchRequest
import com.quickpitik.dto.photographer.HandlePatchRequest
import com.quickpitik.dto.photographer.IncompleteFields
import com.quickpitik.dto.photographer.MediaUploadResponseDto
import com.quickpitik.dto.photographer.RegionPatchRequest
import com.quickpitik.dto.photographer.VerificationSubmitResponseDto
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.ReservedHandleRepository
import com.quickpitik.repository.SocialLinkRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.io.ByteArrayInputStream
import java.time.OffsetDateTime
import java.util.UUID
import javax.imageio.ImageIO

@Service
@Transactional
class PhotographerSettingsService(
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val reservedHandleRepository: ReservedHandleRepository,
    private val regionsService: RegionsService,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val socialLinkRepository: SocialLinkRepository,
    private val payoutAccountRepository: PayoutAccountRepository,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Lazy-create on first /me/photographer/* read. Photographers register via
    // /auth/register with role=PHOTOGRAPHER but the photographer_settings row
    // is created only when they first touch a photographer surface — keeps
    // Phase A unchanged and avoids stale rows for users who never log in.
    fun getOrCreate(userId: UUID): PhotographerSettings {
        val existing = photographerSettingsRepository.findById(userId).orElse(null)
        if (existing != null) return existing
        val user = userRepository.findById(userId).orElseThrow {
            ConflictException(code = "USER_NOT_FOUND", message = "User not found")
        }
        if (user.role != Role.PHOTOGRAPHER) {
            // Method-level @PreAuthorize already enforces this, but defend in
            // depth — if a non-photographer somehow reaches here, refuse to
            // pollute the photographer_settings table with a row.
            throw ConflictException(code = "FORBIDDEN", message = "User is not a photographer")
        }
        val fresh = PhotographerSettings(
            userId = userId,
            verificationStatus = VerificationStatus.INCOMPLETE,
        )
        return photographerSettingsRepository.save(fresh)
    }

    @Transactional(readOnly = true)
    fun findByHandle(handle: String): PhotographerSettings? =
        photographerSettingsRepository.findByHandleIgnoreCase(handle.trim().lowercase())

    // ─── Brand ────────────────────────────────────────────────────────────
    fun putBrand(userId: UUID, req: BrandPatchRequest): PhotographerSettings {
        val settings = getOrCreate(userId)
        req.brandName?.let { name ->
            val trimmed = name.trim()
            if (trimmed.length > 100) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "brandName must be 0-100 chars",
                    field = "brandName",
                )
            }
            settings.brandName = trimmed.ifBlank { null }
        }
        req.brandColor?.let { color ->
            if (color !in ALLOWED_BRAND_COLORS) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "brandColor must be one of $ALLOWED_BRAND_COLORS",
                    field = "brandColor",
                )
            }
            settings.brandColor = color
        }
        req.bio?.let { bio ->
            if (bio.length > 2000) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "bio must be 0-2000 chars",
                    field = "bio",
                )
            }
            settings.bio = bio
        }
        return photographerSettingsRepository.save(settings)
    }

    // ─── Handle ───────────────────────────────────────────────────────────
    fun putHandle(userId: UUID, req: HandlePatchRequest): PhotographerSettings {
        val normalized = req.handle.trim().lowercase()
        if (!HANDLE_REGEX.matches(normalized)) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "handle must be 3-30 chars, lowercase letters/digits/hyphens, no leading/trailing hyphen",
                field = "handle",
            )
        }
        if (reservedHandleRepository.existsByHandle(normalized)) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.RESERVED_HANDLE,
                message = "That handle is reserved.",
                field = "handle",
            )
        }
        val settings = getOrCreate(userId)
        if (settings.handle == normalized) {
            return settings
        }
        // Uniqueness check that excludes self — uses the existing
        // findByHandleIgnoreCase since the column is lowercase by convention.
        val taken = photographerSettingsRepository.findByHandleIgnoreCase(normalized)
        if (taken != null && taken.userId != userId) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.HANDLE_TAKEN,
                message = "That handle is already in use.",
                field = "handle",
            )
        }
        settings.handle = normalized
        return photographerSettingsRepository.save(settings)
    }

    // ─── Region ───────────────────────────────────────────────────────────
    fun putRegion(userId: UUID, req: RegionPatchRequest): PhotographerSettings {
        val regionCode = req.regionCode.trim()
        val provinceCode = req.provinceCode.trim()
        if (!regionsService.isValidPair(regionCode, provinceCode)) {
            throw ValidationException(
                code = ErrorCodes.INVALID_REGION,
                message = "Unknown regionCode/provinceCode pair",
                field = "regionCode",
            )
        }
        val settings = getOrCreate(userId)
        settings.regionCode = regionCode
        settings.provinceCode = provinceCode
        // Mirror provinceCode → city display fallback (province name) so the
        // public profile has something human-readable. PR 9 may swap this for
        // a real city picker; for now province name doubles as locality.
        val regions = regionsService.all()
        val province = regions
            .firstOrNull { it.code == regionCode }
            ?.provinces?.firstOrNull { it.code == provinceCode }
        settings.city = province?.name
        return photographerSettingsRepository.save(settings)
    }

    // ─── Cover ────────────────────────────────────────────────────────────
    fun uploadCover(userId: UUID, bytes: ByteArray, contentType: String?): MediaUploadResponseDto {
        val mime = validateImageMime(contentType, field = "file")
        if (bytes.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "file is empty",
                field = "file",
            )
        }
        val processed = scaleToLongEdgeJpeg(bytes, MAX_COVER_LONG_EDGE)
            ?: throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "image cannot be decoded",
                field = "file",
            )
        val settings = getOrCreate(userId)
        val previousKey = settings.coverS3Key
        val newKey = "photographers/$userId/cover/${UUID.randomUUID()}.jpg"
        storageService.put(newKey, processed, "image/jpeg")
        settings.coverS3Key = newKey
        // Picking an explicit cover image clears the gradient fallback so
        // public-facing renderers don't have to guess which one wins.
        settings.coverGradientFrom = null
        settings.coverGradientTo = null
        photographerSettingsRepository.save(settings)
        if (previousKey != null && previousKey != newKey) {
            runCatching { storageService.delete(previousKey) }
                .onFailure { log.warn("cover delete of {} failed: {}", previousKey, it.message) }
        }
        // Suppress unused mime warning while keeping the validation eager.
        @Suppress("UNUSED_VARIABLE") val _mime = mime
        return MediaUploadResponseDto(
            url = storageService.presignedGetUrl(newKey, storageProperties.presignedTtl.cover),
            uploadedAt = OffsetDateTime.now(),
        )
    }

    // ─── Watermark ────────────────────────────────────────────────────────
    fun uploadWatermark(userId: UUID, bytes: ByteArray, contentType: String?): MediaUploadResponseDto {
        val mime = validateImageMime(contentType, field = "file")
        if (bytes.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "file is empty",
                field = "file",
            )
        }
        val processed = scaleToLongEdgeJpeg(bytes, MAX_WATERMARK_LONG_EDGE)
            ?: throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "image cannot be decoded",
                field = "file",
            )
        val settings = getOrCreate(userId)
        val previousKey = settings.watermarkS3Key
        val newKey = "photographers/$userId/watermark/${UUID.randomUUID()}.jpg"
        storageService.put(newKey, processed, "image/jpeg")
        settings.watermarkS3Key = newKey
        // Photographer hasn't supplied a label slot yet on the FE — derive a
        // safe placeholder from brandName so downstream WatermarkService still
        // has a non-blank label to render onto live photo uploads. The FE may
        // surface a label-edit slab in a later PR; this keeps the upload-only
        // path working today.
        if (settings.watermarkLabel.isNullOrBlank()) {
            settings.watermarkLabel = settings.brandName?.takeIf { it.isNotBlank() }
        }
        photographerSettingsRepository.save(settings)
        if (previousKey != null && previousKey != newKey) {
            runCatching { storageService.delete(previousKey) }
                .onFailure { log.warn("watermark delete of {} failed: {}", previousKey, it.message) }
        }
        @Suppress("UNUSED_VARIABLE") val _mime = mime
        return MediaUploadResponseDto(
            url = storageService.presignedGetUrl(newKey, storageProperties.presignedTtl.watermark),
            uploadedAt = OffsetDateTime.now(),
        )
    }

    // ─── Verification submit ──────────────────────────────────────────────
    // Returns 200 OK in both branches so the FE can read both `status` and
    // `missing` from the typed body. The plan called for 422 + missing[], but
    // the canonical ApiError envelope only carries `code/message/field` and
    // the FE api.ts throws on `success: false` without exposing data — so an
    // incomplete-profile result has to come back as success-shaped data with
    // status="incomplete". Verified state is still PENDING after submit, so
    // there's no ambiguity: if `status` is "incomplete" the FE shows the
    // missing list; if "pending" the FE shows the "submitted" affordance.
    fun submitVerification(userId: UUID): VerificationSubmitResponseDto {
        val settings = getOrCreate(userId)
        val incomplete = collectMissing(userId, settings)
        if (!incomplete.isComplete) {
            return VerificationSubmitResponseDto(
                status = VerificationStatus.INCOMPLETE.toWire(),
                missing = incomplete.missing,
            )
        }
        // PENDING is the post-submit state; admins move it to APPROVED/REJECTED
        // in PR 10. APPROVED/REJECTED rows can be re-submitted (resets to
        // PENDING) so admins see fresh evidence without an out-of-band reset.
        settings.verificationStatus = VerificationStatus.PENDING
        photographerSettingsRepository.save(settings)
        return VerificationSubmitResponseDto(status = settings.verificationStatus.toWire())
    }

    // Used by the controller to pre-build the missing list when an attempt
    // fails so the 422 carries the structured `missing[]` array.
    @Transactional(readOnly = true)
    fun collectMissing(userId: UUID, settings: PhotographerSettings? = null): IncompleteFields {
        val s = settings ?: photographerSettingsRepository.findById(userId).orElse(null)
        val missing = mutableListOf<String>()
        if (s == null || s.regionCode.isNullOrBlank() || s.provinceCode.isNullOrBlank()) {
            missing.add("region")
        }
        if (socialLinkRepository.countByUserId(userId) == 0L) missing.add("socials")
        if (payoutAccountRepository.countByUserId(userId) == 0L) missing.add("payouts")
        return IncompleteFields(missing = missing)
    }

    // ─── Helpers ──────────────────────────────────────────────────────────
    private fun validateImageMime(contentType: String?, field: String): String {
        val mime = (contentType ?: "").lowercase()
        if (mime !in SUPPORTED_IMAGE_TYPES) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "must be jpeg, png, or webp",
                field = field,
            )
        }
        return mime
    }

    // Returns null when ImageIO can't decode the bytes (lying Content-Type).
    private fun scaleToLongEdgeJpeg(input: ByteArray, maxLongEdge: Int): ByteArray? {
        val source = ImageIO.read(ByteArrayInputStream(input)) ?: return null
        val srcW = source.width
        val srcH = source.height
        val longEdge = maxOf(srcW, srcH)
        val scale = if (longEdge > maxLongEdge) maxLongEdge.toDouble() / longEdge else 1.0
        val targetW = (srcW * scale).toInt().coerceAtLeast(1)
        val targetH = (srcH * scale).toInt().coerceAtLeast(1)
        val resized = java.awt.image.BufferedImage(targetW, targetH, java.awt.image.BufferedImage.TYPE_INT_RGB)
        val g = resized.createGraphics()
        try {
            g.setRenderingHint(
                java.awt.RenderingHints.KEY_INTERPOLATION,
                java.awt.RenderingHints.VALUE_INTERPOLATION_BICUBIC,
            )
            g.setRenderingHint(
                java.awt.RenderingHints.KEY_RENDERING,
                java.awt.RenderingHints.VALUE_RENDER_QUALITY,
            )
            g.drawImage(source, 0, 0, targetW, targetH, null)
        } finally {
            g.dispose()
        }
        val out = java.io.ByteArrayOutputStream()
        ImageIO.write(resized, "jpeg", out)
        return out.toByteArray()
    }

    private companion object {
        // Mirrors website/src/store/photographer-settings-store.ts BrandColor.
        val ALLOWED_BRAND_COLORS = setOf("none", "fresh", "amber", "indigo", "rose", "ink")
        val SUPPORTED_IMAGE_TYPES = setOf("image/jpeg", "image/jpg", "image/png", "image/webp")
        // Slug-ish handle: lowercase letters/digits/hyphens, 3-30 chars, no
        // leading/trailing hyphen. Mirrors photographer-settings-store.ts
        // setHandle which lower-cases on the FE side.
        val HANDLE_REGEX = Regex("^[a-z0-9](?:[a-z0-9-]{1,28}[a-z0-9])?$")
        const val MAX_COVER_LONG_EDGE = 1920
        const val MAX_WATERMARK_LONG_EDGE = 1024
    }
}
