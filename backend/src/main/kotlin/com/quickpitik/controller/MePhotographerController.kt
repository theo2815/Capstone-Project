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
import com.quickpitik.dto.photographer.PermanentDeleteResponse
import com.quickpitik.dto.photographer.PhotoActionResponse
import com.quickpitik.dto.photographer.PhotoIdsRequest
import com.quickpitik.dto.photographer.PhotographerBundleDto
import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import com.quickpitik.dto.photographer.UpdateMyEventRequest
import com.quickpitik.dto.photographer.UploadedPhotoDto
import com.quickpitik.dto.photographer.EventMetadataVisibilityDto
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.photographer.PhotoUploadService
import com.quickpitik.service.photographer.PhotographerBundleService
import com.quickpitik.service.photographer.PhotographerEventService
import com.quickpitik.service.photographer.PhotographerOwnedEventService
import com.quickpitik.service.photographer.PhotographerPhotoService
import com.quickpitik.service.photographer.PhotoMetadataSettingsService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import jakarta.servlet.http.HttpServletResponse
import org.springframework.beans.propertyeditors.StringTrimmerEditor
import org.springframework.http.MediaType
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.WebDataBinder
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.InitBinder
import org.springframework.web.bind.annotation.ModelAttribute
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.PutMapping
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/photographer")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerController(
    private val photographerEventService: PhotographerEventService,
    private val photographerOwnedEventService: PhotographerOwnedEventService,
    private val photographerPhotoService: PhotographerPhotoService,
    private val photographerBundleService: PhotographerBundleService,
    private val photoUploadService: PhotoUploadService,
    private val rateLimiter: RateLimiter,
    private val photoMetadataSettingsService: PhotoMetadataSettingsService,
) {
    @GetMapping("/events/{eventId}/metadata-visibility")
    fun getMetadataVisibility(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): EventMetadataVisibilityDto = photoMetadataSettingsService.getEvent(principal.userId, eventId)

    @PutMapping("/events/{eventId}/metadata-visibility")
    fun putMetadataVisibility(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: EventMetadataVisibilityDto,
    ): EventMetadataVisibilityDto =
        photoMetadataSettingsService.setEvent(principal.userId, eventId, body.metadataPublic)

    // ── Photographer-owned events (V46) ───────────────────────────────────
    // multipart/form-data like the admin create: the text parts bind onto the
    // request DTO, the optional `cover` part carries the image bytes.
    @PostMapping(value = ["/events"], consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun createEvent(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @ModelAttribute("form") form: CreateMyEventRequest,
        @RequestPart("cover", required = false) cover: MultipartFile?,
    ): PhotographerEventDetailDto {
        rateLimiter.acquireOrThrow(
            policy = Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD,
            key = principal.userId.toString(),
        )
        return photographerOwnedEventService.create(principal.userId, form, coverOf(cover))
    }

    @PatchMapping(value = ["/events/{eventId}"], consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun updateEvent(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @ModelAttribute("form") form: UpdateMyEventRequest,
        @RequestPart("cover", required = false) cover: MultipartFile?,
    ): PhotographerEventDetailDto =
        photographerOwnedEventService.update(principal.userId, eventId, form, coverOf(cover))

    // FormData sends every field as a string and an untouched input as "", so
    // blank has to read as "absent": the trimmer nulls it, which puts a create
    // back on the DTO default and leaves a patch field unchanged. Scoped by the
    // `form` attribute name to the two event forms. Numbers convert themselves
    // - a non-numeric price is a field error, not a 500.
    @InitBinder("form")
    fun trimEventForm(binder: WebDataBinder) {
        binder.registerCustomEditor(String::class.java, StringTrimmerEditor(true))
    }

    private fun coverOf(cover: MultipartFile?): AdminEventService.CoverUpload? =
        cover?.takeUnless { it.isEmpty }?.let {
            AdminEventService.CoverUpload(bytes = it.bytes, contentType = it.contentType)
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
        @RequestParam(required = false) deleted: Boolean?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotographerLibraryPhotoDto> =
        photographerEventService.listPhotos(
            photographerId = principal.userId,
            eventId = eventId,
            order = order,
            deleted = deleted ?: false,
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

    // Direct-to-storage upload, step 1 of 2 (2026-09-02). Gates + dedup + a
    // presign, no bytes. Begin and commit share the multipart upload policy.
    @PostMapping("/events/{eventId}/photos/direct")
    fun beginDirectUpload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: DirectUploadBeginRequest,
        response: HttpServletResponse,
    ): DirectUploadBeginResponse {
        response.setHeader("Cache-Control", "no-store")
        rateLimiter.acquireOrThrow(
            policy = Bucket4jRateLimiter.POLICY_PHOTOGRAPHER_UPLOAD,
            key = principal.userId.toString(),
        )
        return photoUploadService.beginDirectUpload(
            photographerId = principal.userId,
            eventId = eventId,
            contentHash = body.contentHash,
            contentType = body.contentType,
            sizeBytes = body.sizeBytes,
            requestId = body.requestId,
            client = body.client,
        )
    }

    // Direct-to-storage upload, step 2 of 2: the object is in storage; register it.
    @PostMapping("/events/{eventId}/photos/direct/commit")
    fun commitDirectUpload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: DirectUploadCommitRequest,
        response: HttpServletResponse,
    ): UploadedPhotoDto {
        response.setHeader("Cache-Control", "no-store")
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
            sizeBytes = body.sizeBytes,
            contentType = body.contentType,
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

    // ── Photo library management: trash / restore / permanent delete ───────
    // Bulk by design — a per-photo action in the website/mobile UI sends a
    // single-element list. Ownership + event scoping are enforced in the
    // service (a foreign or wrong-event id is silently skipped).
    @PostMapping("/events/{eventId}/photos/trash")
    fun trashPhotos(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: PhotoIdsRequest,
    ): PhotoActionResponse =
        PhotoActionResponse(photographerPhotoService.trash(principal.userId, eventId, body.photoIds))

    @PostMapping("/events/{eventId}/photos/restore")
    fun restorePhotos(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: PhotoIdsRequest,
    ): PhotoActionResponse =
        PhotoActionResponse(photographerPhotoService.restore(principal.userId, eventId, body.photoIds))

    // Irreversible: deletes the DB row and the R2 objects. Capped lower than
    // trash/restore because each photo is up to 3 single-key R2 deletes. Sold
    // photos are refused and returned in blockedSold (a buyer may still hold a
    // download grant).
    @PostMapping("/events/{eventId}/photos/permanent-delete")
    fun permanentDeletePhotos(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: PhotoIdsRequest,
    ): PermanentDeleteResponse {
        if (body.photoIds.size > PERMANENT_DELETE_BATCH_MAX) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "At most $PERMANENT_DELETE_BATCH_MAX photos per permanent delete",
                field = "photoIds",
            )
        }
        return photographerPhotoService.permanentDelete(principal.userId, eventId, body.photoIds)
    }

    // "Download All" step 1: mint a short-lived tokened URL the client then
    // navigates to (the ZIP stream is token-auth — a top-level <a> nav carries
    // no JWT). Rate-limited like the order bundle.
    @PostMapping("/events/{eventId}/download-bundle")
    fun prepareBundle(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): PhotographerBundleDto {
        rateLimiter.acquireOrThrow(
            policy = Bucket4jRateLimiter.POLICY_BUNDLE_DOWNLOAD,
            key = principal.userId.toString(),
        )
        return photographerBundleService.prepareDownloadUrl(principal.userId, eventId)
    }

    private companion object {
        const val PERMANENT_DELETE_BATCH_MAX = 100
    }
}
