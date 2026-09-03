package com.quickpitik.service.photos

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photos.PhotoVerifyResultDto
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.image.ExifOrientation
import com.quickpitik.service.image.ImagePixelGuard
import com.quickpitik.service.image.PerceptualHash
import org.springframework.beans.factory.annotation.Value
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import java.io.ByteArrayInputStream
import java.util.UUID
import javax.imageio.ImageIO

// "Is this a QuickPitik photo, and whose?" — fingerprints the uploaded image
// the same way PhotoWatermarkService fingerprints every preview, then asks the
// registry for the nearest stored hash. The photographer name follows the same
// rule as the baked credit: brand name, else account name.
@Service
class PhotoVerifyService(
    private val photoRepository: PhotoRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val eventRepository: EventRepository,
    @Value("\${app.watermark.verify-max-distance:12}") private val maxDistance: Int,
) {
    private val log = org.slf4j.LoggerFactory.getLogger(javaClass)

    // Not @Transactional: decode + hash is CPU work that must not pin a
    // connection; the three PK reads afterwards are independent.
    fun verify(bytes: ByteArray): PhotoVerifyResultDto {
        if (ImagePixelGuard.exceedsPixelBudget(bytes)) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Image is too large to fingerprint",
                field = "file",
            )
        }
        val decoded = ImageIO.read(ByteArrayInputStream(bytes))
            ?: throw ApiException(
                status = HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "file must be a decodable JPEG or PNG image",
                field = "file",
            )
        val hash = PerceptualHash.of(ExifOrientation.apply(decoded, ExifOrientation.read(bytes)))

        val row = photoRepository.findNearestByPhash(hash).firstOrNull() ?: return PhotoVerifyResultDto.NONE
        val distance = (row[2] as Number).toInt()
        // The exact distance stays server-side: on the wire it would be an
        // oracle for editing a copy until it stops matching.
        log.debug("photo-verify nearest distance {} (threshold {})", distance, maxDistance)
        if (distance > maxDistance) return PhotoVerifyResultDto.NONE

        // photographer_id is nullable (legacy/seed rows) — a match with no
        // owner still confirms "this is a QuickPitik photo".
        val photographerId = row[0] as? UUID
        val settings = photographerId?.let { photographerSettingsRepository.findById(it).orElse(null) }
        val user = photographerId?.let { userRepository.findById(it).orElse(null) }
        val event = (row[1] as? UUID)?.let { eventRepository.findById(it).orElse(null) }
        return PhotoVerifyResultDto(
            matched = true,
            confidence = if (distance <= maxDistance / 2) "strong" else "weak",
            photographerName = settings?.brandName?.takeIf { it.isNotBlank() } ?: user?.name,
            photographerHandle = settings?.handle,
            eventName = event?.name,
            eventDate = event?.date,
        )
    }
}
