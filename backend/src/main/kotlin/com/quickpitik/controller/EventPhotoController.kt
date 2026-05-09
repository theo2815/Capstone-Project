package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.photos.PhotoService
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/api/v1/events")
class EventPhotoController(
    private val eventRepository: EventRepository,
    private val photoService: PhotoService,
    private val photoSearchService: PhotoSearchService,
) {
    @GetMapping("/{slug}/photos")
    fun list(
        @PathVariable slug: String,
        @RequestParam(required = false) bib: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotoDto> {
        val event = eventRepository.findBySlugAndDeletedAtIsNull(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        return photoService.listForEvent(event.id, bib, PaginationParams.of(offset, limit))
    }

    @PostMapping(
        value = ["/{slug}/photos/search-by-face"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
    )
    fun searchByFaceMultipart(
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
        )
    }

    @PostMapping(
        value = ["/{slug}/photos/search-by-face"],
        consumes = [MediaType.APPLICATION_JSON_VALUE],
    )
    fun searchByFaceJson(
        @PathVariable slug: String,
        @RequestBody body: SearchByFaceJsonRequest,
    ): PaginatedResponse<PhotoDto> {
        // Resolving a stored selfieId requires the user_selfies table that
        // ships in PR 6 (Profile). Until then we surface a clear error so the
        // FE can fall back to the multipart upload path.
        throw ValidationException(
            code = ErrorCodes.SELFIE_REQUIRED,
            message = "Stored selfies are not available yet — upload a selfie file with the search request.",
            field = "selfieId",
        )
    }

    data class SearchByFaceJsonRequest(
        val selfieId: String? = null,
        val offset: Int? = null,
        val limit: Int? = null,
    )
}
