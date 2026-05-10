package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.ImageIO

/**
 * POST /me/avatar uploads, normalises (512×512 center-crop, JPEG re-encode),
 * and stores the avatar. Mirrors the FE's `lib/image-utils.ts` so the same
 * normalisation runs whether the FE pre-cropped or not — clients lie about
 * dimensions, the server is the source of truth.
 *
 * The previous avatar's S3 object is best-effort deleted; failure to delete
 * doesn't block the upload because the new key is already persisted.
 */
@Service
@Transactional
class AvatarService(
    private val userRepository: UserRepository,
    private val storageService: StorageService,
    private val userDtoMapper: UserDtoMapper,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun upload(userId: UUID, file: ByteArray, contentType: String?): UserDto {
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
                message = "avatar must be jpeg, png, or webp",
                field = "file",
            )
        }
        val processed = centerCropToJpeg(file)
        val user = userRepository.findById(userId).orElseThrow {
            UnauthorizedException(code = ErrorCodes.UNAUTHORIZED, message = "User not found")
        }
        val previousKey = user.avatarS3Key
        val newKey = "avatars/$userId/${UUID.randomUUID()}.jpg"
        storageService.put(newKey, processed, "image/jpeg")
        user.avatarS3Key = newKey
        userRepository.save(user)
        if (previousKey != null && previousKey != newKey) {
            runCatching { storageService.delete(previousKey) }
                .onFailure { log.warn("avatar delete of previous key {} failed: {}", previousKey, it.message) }
        }
        return userDtoMapper.toDto(user)
    }

    fun delete(userId: UUID): UserDto {
        val user = userRepository.findById(userId).orElseThrow {
            UnauthorizedException(code = ErrorCodes.UNAUTHORIZED, message = "User not found")
        }
        val previousKey = user.avatarS3Key
        user.avatarS3Key = null
        userRepository.save(user)
        if (previousKey != null) {
            runCatching { storageService.delete(previousKey) }
                .onFailure { log.warn("avatar delete of {} failed: {}", previousKey, it.message) }
        }
        return userDtoMapper.toDto(user)
    }

    private fun centerCropToJpeg(input: ByteArray): ByteArray {
        val original: BufferedImage = ImageIO.read(ByteArrayInputStream(input))
            ?: throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "image cannot be decoded",
                field = "file",
            )
        val w = original.width
        val h = original.height
        val crop = minOf(w, h)
        val x = (w - crop) / 2
        val y = (h - crop) / 2
        val cropped = original.getSubimage(x, y, crop, crop)
        val output = BufferedImage(SIDE, SIDE, BufferedImage.TYPE_INT_RGB)
        val g = output.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
            g.drawImage(cropped, 0, 0, SIDE, SIDE, null)
        } finally {
            g.dispose()
        }
        val out = ByteArrayOutputStream()
        ImageIO.write(output, "jpeg", out)
        return out.toByteArray()
    }

    private companion object {
        const val SIDE = 512
        val SUPPORTED_TYPES = setOf("image/jpeg", "image/jpg", "image/png", "image/webp")
    }
}
