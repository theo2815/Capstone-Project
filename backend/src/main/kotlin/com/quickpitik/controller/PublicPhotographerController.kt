package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photographer.PhotographerProfileDto
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.service.photographer.PublicPhotographerService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.servlet.http.HttpServletRequest
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/public/photographers")
class PublicPhotographerController(
    private val publicPhotographerService: PublicPhotographerService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping("/{handle}")
    fun getProfile(
        @PathVariable handle: String,
        request: HttpServletRequest,
    ): PhotographerProfileDto {
        rateLimit(request)
        return publicPhotographerService.getProfile(handle)
    }

    @GetMapping("/{handle}/events/{slug}/photos")
    fun listEventPhotos(
        @PathVariable handle: String,
        @PathVariable slug: String,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
        request: HttpServletRequest,
    ): PaginatedResponse<PhotoDto> {
        rateLimit(request)
        return publicPhotographerService.listEventPhotos(
            handle = handle,
            eventSlug = slug,
            pagination = PaginationParams.of(offset, limit),
        )
    }

    private fun rateLimit(request: HttpServletRequest) {
        rateLimiter.acquireOrThrow(
            policy = Bucket4jRateLimiter.POLICY_PUBLIC_GALLERY,
            key = clientIp(request),
        )
    }

    // X-Forwarded-For wins when present (rendered behind a load balancer); use
    // the first hop and fall back to the direct remote address for local dev.
    private fun clientIp(request: HttpServletRequest): String {
        val forwarded = request.getHeader("X-Forwarded-For")
        if (!forwarded.isNullOrBlank()) {
            return forwarded.split(",").first().trim()
        }
        return request.remoteAddr ?: "unknown"
    }
}
