package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.photos.PhotoService
import com.quickpitik.service.storage.StorageService
import org.springframework.http.MediaType
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@RestController
@RequestMapping("/api/v1/events")
class EventPhotoController(
    private val eventRepository: EventRepository,
    private val photoService: PhotoService,
    private val photoSearchService: PhotoSearchService,
    private val userSelfieRepository: UserSelfieRepository,
    private val storageService: StorageService,
) {
    @GetMapping("/{slug}/photos")
    fun list(
        @AuthenticationPrincipal principal: AuthPrincipal?,
        @PathVariable slug: String,
        @RequestParam(required = false) bib: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotoDto> {
        val event = eventRepository.findBySlugAndDeletedAtIsNull(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        return photoService.listForEvent(
            eventId = event.id,
            bib = bib,
            pagination = PaginationParams.of(offset, limit),
            requesterUserId = principal?.userId,
        )
    }

    @PostMapping(
        value = ["/{slug}/photos/search-by-face"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
    )
    fun searchByFaceMultipart(
        @AuthenticationPrincipal principal: AuthPrincipal?,
        @PathVariable slug: String,
        @RequestPart("selfie") selfie: MultipartFile,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotoDto> {
        val event = eventRepository.findBySlugAndDeletedAtIsNull(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        return photoSearchService.searchByFace(
            eventId = event.id,
            selfieBytes = selfie.bytes,
            contentType = selfie.contentType ?: MediaType.APPLICATION_OCTET_STREAM_VALUE,
            filename = selfie.originalFilename ?: "selfie",
            pagination = PaginationParams.of(offset, limit),
            requesterUserId = principal?.userId,
        )
    }

    // Stored-selfie path (G-1, 2026-05-19 PM). The runner picks a selfie from
    // their library (typically the primary), the FE passes the selfieId, and
    // we hand the stored bytes off to the same PhotoSearchService used by the
    // multipart path. Auth-required: the selfie is scoped to the calling user
    // so cross-user IDOR is impossible — `findByIdAndUserId` only returns the
    // row when both match. AI_API_ENABLED=false short-circuits inside
    // PhotoSearchService (503), matching the multipart path.
    @PostMapping(
        value = ["/{slug}/photos/search-by-face"],
        consumes = [MediaType.APPLICATION_JSON_VALUE],
    )
    fun searchByFaceJson(
        @AuthenticationPrincipal principal: AuthPrincipal?,
        @PathVariable slug: String,
        @RequestBody body: SearchByFaceJsonRequest,
    ): PaginatedResponse<PhotoDto> {
        if (principal == null) {
            throw UnauthorizedException(
                code = ErrorCodes.UNAUTHORIZED,
                message = "Sign in to search with a stored selfie.",
            )
        }
        val rawId = body.selfieId?.trim().orEmpty()
        if (rawId.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.SELFIE_REQUIRED,
                message = "selfieId is required",
                field = "selfieId",
            )
        }
        val selfieUuid = runCatching { UUID.fromString(rawId) }.getOrNull()
            ?: throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "selfieId must be a UUID",
                field = "selfieId",
            )
        val event = eventRepository.findBySlugAndDeletedAtIsNull(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val selfie = userSelfieRepository.findByIdAndUserId(selfieUuid, principal.userId)
            ?: throw NotFoundException(
                code = ErrorCodes.SELFIE_NOT_FOUND,
                message = "Selfie not found",
            )
        val bytes = storageService.getBytes(selfie.s3Key)
        return photoSearchService.searchByFace(
            eventId = event.id,
            selfieBytes = bytes,
            contentType = contentTypeOf(selfie.s3Key),
            filename = selfie.s3Key.substringAfterLast('/'),
            pagination = PaginationParams.of(body.offset, body.limit),
            requesterUserId = principal.userId,
        )
    }

    private fun contentTypeOf(key: String): String = when (key.substringAfterLast('.').lowercase()) {
        "jpg", "jpeg" -> MediaType.IMAGE_JPEG_VALUE
        "png" -> MediaType.IMAGE_PNG_VALUE
        "webp" -> "image/webp"
        else -> MediaType.APPLICATION_OCTET_STREAM_VALUE
    }

    data class SearchByFaceJsonRequest(
        val selfieId: String? = null,
        val offset: Int? = null,
        val limit: Int? = null,
    )
}
