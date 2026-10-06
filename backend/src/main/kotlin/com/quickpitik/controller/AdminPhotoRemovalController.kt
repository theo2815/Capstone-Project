package com.quickpitik.controller

import com.quickpitik.dto.removal.*
import com.quickpitik.entity.RemovalRequestStatus
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.removal.*
import jakarta.servlet.http.HttpServletResponse
import jakarta.validation.Valid
import org.springframework.http.MediaType
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.*
import java.util.UUID

@RestController
@RequestMapping("/api/v1/admin/photo-removal-requests")
@PreAuthorize("hasRole('ADMIN')")
class AdminPhotoRemovalController(
    private val service: PhotoRemovalService,
    private val queries: PhotoRemovalQueryService,
    private val evidence: PhotoRemovalEvidenceService,
    private val rateLimiter: RateLimiter,
) {
    @PostMapping
    fun create(@AuthenticationPrincipal principal: AuthPrincipal, @Valid @RequestBody command: CreateRemovalRequestCommand,
        response: HttpServletResponse): PhotoRemovalRequestDto {
        limit(principal, response, Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD)
        return service.create(principal.userId, command)
    }

    @GetMapping
    fun list(@AuthenticationPrincipal principal: AuthPrincipal, @RequestParam(required = false) eventId: UUID?,
        @RequestParam(required = false) status: RemovalRequestStatus?, @RequestParam(required = false) needsAction: Boolean?,
        @RequestParam(required = false) offset: Int?, @RequestParam(required = false) limit: Int?, response: HttpServletResponse): RemovalPage<PhotoRemovalRequestDto> {
        limit(principal, response)
        return queries.list(principal.userId, eventId, status, needsAction, offset, limit)
    }

    @GetMapping("/{id}")
    fun detail(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID, response: HttpServletResponse): PhotoRemovalRequestDto {
        limit(principal, response)
        return queries.adminDetail(principal.userId, id)
    }

    @GetMapping("/{id}/history")
    fun history(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID, @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?, response: HttpServletResponse): RemovalPage<RemovalHistoryDto> {
        limit(principal, response)
        return queries.history(principal.userId, id, offset, limit)
    }

    @PutMapping("/{id}/evidence", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun replace(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        @Valid @ModelAttribute form: RemovalEvidenceUploadForm, response: HttpServletResponse): PhotoRemovalRequestDto {
        limit(principal, response, Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD)
        val file = form.selfie!!
        return evidence.replace(principal.userId, id, form.commandId!!, form.expectedVersion!!,
            SearchImage(file.bytes, file.contentType ?: "", file.originalFilename ?: "selfie"))
    }

    @GetMapping("/{id}/evidence")
    fun stream(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        response: HttpServletResponse) {
        limit(principal, response)
        val authorized = evidence.stream(principal.userId, id)
        response.contentType = authorized.contentType
        response.setHeader("X-Content-Type-Options", "nosniff")
        // The bounded evidence stream finishes in this authenticated request. No ASYNC redispatch.
        authorized.stream.use { it.copyTo(response.outputStream) }
    }

    @PostMapping("/{id}/cancel")
    fun cancel(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        @Valid @RequestBody command: CancelRemovalCommand, response: HttpServletResponse): PhotoRemovalRequestDto {
        limit(principal, response, Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD)
        return service.cancel(principal.userId, id, command)
    }

    @GetMapping("/{id}/photos")
    fun photos(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        @RequestParam(required = false) bib: String?, @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?, response: HttpServletResponse): RemovalPage<RemovalPhotoDto> {
        limit(principal, response, Bucket4jRateLimiter.POLICY_PHOTO_SEARCH)
        return queries.photos(principal.userId, id, bib, offset, limit)
    }

    @PostMapping("/{id}/matches")
    fun matches(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        @Valid @RequestBody command: MatchRemovalCommand, response: HttpServletResponse): RemovalMatchResult {
        limit(principal, response, Bucket4jRateLimiter.POLICY_PHOTO_SEARCH)
        return queries.matches(principal.userId, id, command)
    }

    @PostMapping("/{id}/verify")
    fun verify(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        @Valid @RequestBody command: VerifyRemovalCommand, response: HttpServletResponse): PhotoRemovalRequestDto {
        limit(principal, response, Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD)
        return service.verify(principal.userId, id, command)
    }

    private fun limit(principal: AuthPrincipal, response: HttpServletResponse, policy: String = Bucket4jRateLimiter.POLICY_ORDER_READ) {
        response.setHeader("Cache-Control", "private, no-store")
        rateLimiter.acquireOrThrow(policy, principal.userId.toString())
    }
}
