package com.quickpitik.controller

import com.quickpitik.dto.profile.SelfieRefDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.profile.SelfieService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import org.springframework.http.MediaType
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/selfies")
class MeSelfieController(
    private val selfieService: SelfieService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping
    fun list(@AuthenticationPrincipal principal: AuthPrincipal): List<SelfieRefDto> =
        selfieService.list(principal.userId)

    @PostMapping(consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun upload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestPart("file") file: MultipartFile,
    ): SelfieRefDto {
        // Selfie uploads can trigger AI quality inference — per-user throttle.
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, principal.userId.toString())
        return selfieService.upload(
            userId = principal.userId,
            file = file.bytes,
            contentType = file.contentType,
            filename = file.originalFilename,
        )
    }

    @DeleteMapping("/{id}")
    fun delete(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): Map<String, Boolean> =
        mapOf("removed" to selfieService.delete(principal.userId, id))

    @PostMapping("/{id}/set-primary")
    fun setPrimary(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): List<SelfieRefDto> = selfieService.setPrimary(principal.userId, id)
}
