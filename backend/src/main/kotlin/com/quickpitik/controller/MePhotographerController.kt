package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photographer.PhotographerDownloadDto
import com.quickpitik.dto.photographer.PhotographerEventDetailDto
import com.quickpitik.dto.photographer.PhotographerEventSummaryDto
import com.quickpitik.dto.photographer.DirectUploadBeginRequest
import com.quickpitik.dto.photographer.DirectUploadBeginResponse
import com.quickpitik.dto.photographer.DirectUploadCommitRequest
import com.quickpitik.dto.photographer.CreateMyEventRequest
import com.quickpitik.dto.photographer.PhotoExistsRequest
import com.quickpitik.dto.photographer.PhotoExistsResponse
import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import com.quickpitik.dto.photographer.UpdateMyEventRequest
import com.quickpitik.dto.photographer.UploadedPhotoDto
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.photographer.PhotoUploadService
import com.quickpitik.service.photographer.PhotographerEventService
import com.quickpitik.service.photographer.PhotographerOwnedEventService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import org.springframework.http.MediaType
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.math.BigDecimal
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/photographer")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerController(
    private val photographerEventService: PhotographerEventService,
    private val photographerOwnedEventService: PhotographerOwnedEventService,
    private val photoUploadService: PhotoUploadService,
    private val rateLimiter: RateLimiter,
) {
    // ── Photographer-owned events (V46) ───────────────────────────────────
    // multipart/form-data like the admin create: text fields as @RequestParam,
    // the optional `cover` part carries the image bytes.
    @PostMapping(value = ["/events"], consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun createEvent(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam("title") title: String,
        @RequestParam("date") date: String,
        @RequestParam("location") location: String,
        @RequestParam("organizerName", required = false) organizerName: String?,
        @RequestParam("description", required = false) description: String?,
        @RequestParam("visibility", required = false) visibility: String?,
        @RequestParam("pricingMode", required = false) pricingMode: String?,
        @RequestParam("pricePerPhoto", required = false) pricePerPhoto: String?,
        @RequestParam("watermarkPolicy", required = false) watermarkPolicy: String?,
        @RequestPart("cover", required = false) cover: MultipartFile?,
    ): PhotographerEventDetailDto {
        rateLimiter.acquireOrThrow(
            policy = Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD,
            key = principal.userId.toString(),
        )
        val req = CreateMyEventRequest(
            title = title.trim(),
            date = date.trim(),
            location = location.trim(),
            organizerName = organizerName?.trim()?.takeIf { it.isNotEmpty() },
            description = description?.trim()?.takeIf { it.isNotEmpty() },
            visibility = visibility?.trim()?.takeIf { it.isNotEmpty() } ?: "public",
            pricingMode = pricingMode?.trim()?.takeIf { it.isNotEmpty() } ?: "paid",
            pricePerPhoto = parsePrice(pricePerPhoto),
            watermarkPolicy = watermarkPolicy?.trim()?.takeIf { it.isNotEmpty() },
        )
        if (req.title.isBlank()) {
            throw ValidationException(code = ErrorCodes.VALIDATION_ERROR, message = "title is required", field = "title")
        }
        if (req.date.isBlank()) {
            throw ValidationException(code = ErrorCodes.VALIDATION_ERROR, message = "date is required", field = "date")
        }
        if (req.location.isBlank()) {
            throw ValidationException(code = ErrorCodes.VALIDATION_ERROR, message = "location is required", field = "location")
        }
        return photographerOwnedEventService.create(principal.userId, req, coverOf(cover))
    }

    @PatchMapping(value = ["/events/{eventId}"], consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun updateEvent(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @RequestParam(required = false) title: String?,
        @RequestParam(required = false) date: String?,
        @RequestParam(required = false) location: String?,
        @RequestParam(required = false) organizerName: String?,
        @RequestParam(required = false) description: String?,
        @RequestParam(required = false) visibility: String?,
        @RequestParam(required = false) pricingMode: String?,
        @RequestParam(required = false) pricePerPhoto: String?,
        @RequestParam(required = false) watermarkPolicy: String?,
        @RequestParam(required = false) withdrawPendingChange: Boolean?,
        @RequestPart("cover", required = false) cover: MultipartFile?,
    ): PhotographerEventDetailDto {
        val req = UpdateMyEventRequest(
            title = title?.trim()?.takeIf { it.isNotEmpty() },
            date = date?.trim()?.takeIf { it.isNotEmpty() },
            location = location?.trim()?.takeIf { it.isNotEmpty() },
            organizerName = organizerName?.trim()?.takeIf { it.isNotEmpty() },
            description = description?.trim()?.takeIf { it.isNotEmpty() },
            visibility = visibility?.trim()?.takeIf { it.isNotEmpty() },
            pricingMode = pricingMode?.trim()?.takeIf { it.isNotEmpty() },
            pricePerPhoto = parsePrice(pricePerPhoto),
            watermarkPolicy = watermarkPolicy?.trim()?.takeIf { it.isNotEmpty() },
            withdrawPendingChange = withdrawPendingChange == true,
        )
        return photographerOwnedEventService.update(principal.userId, eventId, req, coverOf(cover))
    }

    private fun coverOf(cover: MultipartFile?): AdminEventService.CoverUpload? =
        cover?.takeUnless { it.isEmpty }?.let {
            AdminEventService.CoverUpload(bytes = it.bytes, contentType = it.contentType)
        }

    // FormData round-trips numbers as strings; blank = not supplied.
    private fun parsePrice(raw: String?): BigDecimal? {
        val trimmed = raw?.trim().orEmpty()
        if (trimmed.isEmpty()) return null
        return runCatching { BigDecimal(trimmed) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "pricePerPhoto must be a number",
                field = "pricePerPhoto",
            )
        }
    }

    @GetMapping("/events")
    fun listEvents(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(required = false) withUploads: Boolean?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotographerEventSummaryDto> =
        photographerEventService.listEvents(
            photographerId = principal.userId,
            withUploads = withUploads ?: false,
            pagination = PaginationParams.of(offset, limit),
        )

    @GetMapping("/events/{eventId}")
    fun getEvent(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): PhotographerEventDetailDto =
        photographerEventService.getEventDetail(
            photographerId = principal.userId,
            eventId = eventId,
        )

    @GetMapping("/events/{eventId}/photos")
    fun listEventPhotos(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @RequestParam(required = false) order: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotographerLibraryPhotoDto> =
        photographerEventService.listPhotos(
            photographerId = principal.userId,
            eventId = eventId,
            order = order,
            pagination = PaginationParams.of(offset, limit),
        )

    @GetMapping("/photos/{photoId}/download")
    fun getDownload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable photoId: UUID,
    ): PhotographerDownloadDto =
        photographerEventService.getDownload(
            photographerId = principal.userId,
            photoId = photoId,
        )

    @PostMapping(
        value = ["/events/{eventId}/photos"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
    )
    fun uploadPhoto(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @RequestPart("file") file: MultipartFile,
    ): UploadedPhotoDto {
        rateLimiter.acquireOrThrow(
            policy = Bucket4jRateLimiter.POLICY_PHOTOGRAPHER_UPLOAD,
            key = principal.userId.toString(),
        )
        return photoUploadService.upload(
            photographerId = principal.userId,
            eventId = eventId,
            file = file,
        )
    }

    // Direct-to-storage upload, step 1 of 2 (2026-09-02). Cheap: gates + dedup
    // + a presign, no bytes. The rate-limited step is the commit, where the
    // photo actually becomes a row — same policy as the multipart upload.
    @PostMapping("/events/{eventId}/photos/direct")
    fun beginDirectUpload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: DirectUploadBeginRequest,
    ): DirectUploadBeginResponse =
        photoUploadService.beginDirectUpload(
            photographerId = principal.userId,
            eventId = eventId,
            contentHash = body.contentHash,
            contentType = body.contentType,
            sizeBytes = body.sizeBytes,
        )

    // Direct-to-storage upload, step 2 of 2: the object is in storage; register it.
    @PostMapping("/events/{eventId}/photos/direct/commit")
    fun commitDirectUpload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: DirectUploadCommitRequest,
    ): UploadedPhotoDto {
        rateLimiter.acquireOrThrow(
            policy = Bucket4jRateLimiter.POLICY_PHOTOGRAPHER_UPLOAD,
            key = principal.userId.toString(),
        )
        return photoUploadService.commitDirectUpload(
            photographerId = principal.userId,
            eventId = eventId,
            photoId = body.photoId,
            key = body.key,
            contentHash = body.contentHash,
        )
    }

    // Pre-flight duplicate check (dedup Phase 2). The client hashes its files
    // locally and asks which are already present before uploading, so it can
    // skip re-sending bytes the photographer already has. POST (not GET) because
    // the hash list is a body, not a query string. Read-only and authenticated;
    // like the other photographer read endpoints it carries no rate-limit
    // policy — the actual upload is where the cost (and the limit) lives.
    @PostMapping("/events/{eventId}/photos/exists")
    fun checkPhotosExist(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: PhotoExistsRequest,
    ): PhotoExistsResponse =
        photoUploadService.checkExisting(
            photographerId = principal.userId,
            eventId = eventId,
            hashes = body.hashes,
        )
}
