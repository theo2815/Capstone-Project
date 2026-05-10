package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photographer.PhotographerDownloadDto
import com.quickpitik.dto.photographer.PhotographerEventDetailDto
import com.quickpitik.dto.photographer.PhotographerEventSummaryDto
import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import com.quickpitik.dto.photographer.UploadedPhotoDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.photographer.PhotoUploadService
import com.quickpitik.service.photographer.PhotographerEventService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import org.springframework.http.MediaType
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/photographer")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerController(
    private val photographerEventService: PhotographerEventService,
    private val photoUploadService: PhotoUploadService,
    private val rateLimiter: RateLimiter,
) {
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
}
