package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photos.PhotoVerifyResultDto
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.photos.PhotoVerifyService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.ratelimit.clientIp
import jakarta.servlet.http.HttpServletRequest
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

// Public: anyone holding a screenshot may ask whose photo it is. IP-keyed
// bucket because there is no principal. The answer is attribution only (see
// PhotoVerifyResultDto), so probing the registry reveals nothing a public
// gallery doesn't already show. Same upload gate shape as the guest face
// search in EventPhotoController.
@RestController
@RequestMapping("/api/v1/public/photos")
class PublicPhotoVerifyController(
    private val photoVerifyService: PhotoVerifyService,
    private val rateLimiter: RateLimiter,
) {
    @PostMapping("/verify", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun verify(
        @RequestPart("file") file: MultipartFile,
        request: HttpServletRequest,
    ): PhotoVerifyResultDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PHOTO_VERIFY, clientIp(request))
        if (file.isEmpty) {
            throw ValidationException(code = ErrorCodes.VALIDATION_ERROR, message = "file is required", field = "file")
        }
        val mime = (file.contentType ?: "").lowercase().substringBefore(';').trim()
        if (mime !in SUPPORTED_TYPES) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "file must be jpeg or png",
                field = "file",
            )
        }
        if (file.size > MAX_BYTES) {
            throw ApiException(
                status = HttpStatus.PAYLOAD_TOO_LARGE,
                code = ErrorCodes.PAYLOAD_TOO_LARGE,
                message = "File must be ≤ ${MAX_BYTES / (1024 * 1024)} MB",
                field = "file",
            )
        }
        return photoVerifyService.verify(file.bytes)
    }

    private companion object {
        const val MAX_BYTES = 10L * 1024 * 1024
        // ImageIO decodes JPEG + PNG only; a WebP screenshot would 415 at decode.
        val SUPPORTED_TYPES = setOf("image/jpeg", "image/jpg", "image/png")
    }
}
