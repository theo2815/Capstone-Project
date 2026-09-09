package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.NotFoundException
import com.quickpitik.service.photographer.PhotographerBundleService
import com.quickpitik.service.photographer.PhotographerBundleTokenService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.ratelimit.clientIp
import jakarta.servlet.http.HttpServletRequest
import org.springframework.http.HttpHeaders
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.UUID

// Photographer "Download All" ZIP. Token-only (not JWT) because a top-level
// <a> navigation carries no Authorization header — same reason and shape as
// GuestOrderController's bundle route. SecurityConfig permits this path; the
// signed token (minted by MePhotographerController.prepareBundle) is the
// credential, and it only ever grants the caller's own photos in one event.
//
// StreamingResponseBody bypasses ResponseEnvelopeAdvice, so the binary stream
// isn't wrapped in the JSON envelope.
@RestController
@RequestMapping("/api/v1/photographer-bundles")
class PhotographerBundleController(
    private val bundleService: PhotographerBundleService,
    private val tokenService: PhotographerBundleTokenService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping("/{eventId}")
    fun download(
        @PathVariable eventId: UUID,
        @RequestParam(required = false) token: String?,
        request: HttpServletRequest,
    ): ResponseEntity<StreamingResponseBody> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_BUNDLE_DOWNLOAD, clientIp(request))
        // Anti-IDOR: any bad/expired/wrong-event token is a uniform NOT_FOUND.
        val photographerId = tokenService.verify(token, eventId)
            ?: throw NotFoundException(ErrorCodes.PHOTO_NOT_FOUND, "Not found")
        val spec = bundleService.prepare(photographerId, eventId)
        val encoded = URLEncoder.encode(spec.filename, StandardCharsets.UTF_8).replace("+", "%20")
        val body = StreamingResponseBody { out -> bundleService.writeTo(spec, out) }
        return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_TYPE, spec.contentType)
            .header(
                HttpHeaders.CONTENT_DISPOSITION,
                """attachment; filename="${spec.filename}"; filename*=UTF-8''$encoded""",
            )
            .header(HttpHeaders.CACHE_CONTROL, "no-store")
            .body(body)
    }
}
