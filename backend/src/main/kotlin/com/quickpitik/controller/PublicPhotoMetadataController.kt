package com.quickpitik.controller

import com.quickpitik.dto.photos.PhotoMetadataDto
import com.quickpitik.service.photographer.PhotoMetadataService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.ratelimit.clientIp
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

@RestController
@RequestMapping("/api/v1/public/photos")
class PublicPhotoMetadataController(
    private val metadata: PhotoMetadataService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping("/{photoId}/metadata")
    fun get(
        @PathVariable photoId: UUID,
        request: HttpServletRequest,
        response: HttpServletResponse,
    ): PhotoMetadataDto {
        response.setHeader("Cache-Control", "private, no-store")
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PUBLIC_GALLERY, clientIp(request))
        return metadata.getPublic(photoId)
    }
}
