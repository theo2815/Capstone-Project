package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.photos.PhotoService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.ratelimit.clientIp
import com.quickpitik.service.storage.StorageService
import jakarta.servlet.http.HttpServletRequest
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
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
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.time.OffsetDateTime
import java.util.UUID

@RestController
@RequestMapping("/api/v1/events")
class EventPhotoController(
    private val eventRepository: EventRepository,
    private val photoService: PhotoService,
    private val photoSearchService: PhotoSearchService,
    private val userSelfieRepository: UserSelfieRepository,
    private val storageService: StorageService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping("/{slug}/photos")
    fun list(
        @AuthenticationPrincipal principal: AuthPrincipal?,
        @PathVariable slug: String,
        @RequestParam(required = false) bib: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
        @RequestParam(required = false) snapshotAt: OffsetDateTime?,
        request: HttpServletRequest,
    ): PaginatedResponse<PhotoDto> {
        // Throttle the *search* use of this endpoint only. Without a bib this
        // is the plain event grid that every visitor pages through, and a
        // 10/min cap would break ordinary browsing.
        if (!bib.isNullOrBlank()) {
            rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PHOTO_SEARCH, searchKey(principal, request))
        }
        val event = eventRepository.findPublicBySlug(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        return photoService.listForEvent(
            eventId = event.id,
            bib = bib,
            pagination = PaginationParams.of(offset, limit),
            requesterUserId = principal?.userId,
            snapshotAt = snapshotAt,
        )
    }

    // Free-event original, streamed through the backend (2026-09-05). A
    // presigned R2 URL handed to a top-level navigation gets `fbclid=`
    // appended by Meta's in-app browsers and fails its SigV4 check; this
    // route ignores unknown params. Public: the gate is the event's pricing
    // mode, enforced in PhotoService.freeDownload. StreamingResponseBody
    // bypasses ResponseEnvelopeAdvice (same as the order bundle).
    @GetMapping("/{eventId}/photos/{photoId}/download")
    fun downloadFree(
        @PathVariable eventId: UUID,
        @PathVariable photoId: UUID,
        request: HttpServletRequest,
    ): ResponseEntity<StreamingResponseBody> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PHOTO_DOWNLOAD, clientIp(request))
        val download = photoService.freeDownload(eventId, photoId)
        val encoded = URLEncoder.encode(download.filename, StandardCharsets.UTF_8).replace("+", "%20")
        val body = StreamingResponseBody { out -> storageService.open(download.s3Key).use { it.transferTo(out) } }
        return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_TYPE, MediaType.IMAGE_JPEG_VALUE)
            .header(
                HttpHeaders.CONTENT_DISPOSITION,
                """attachment; filename="${download.filename}"; filename*=UTF-8''$encoded""",
            )
            .header(HttpHeaders.CACHE_CONTROL, "no-store")
            .body(body)
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
        request: HttpServletRequest,
    ): PaginatedResponse<PhotoDto> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PHOTO_SEARCH, searchKey(principal, request))
        validateSelfieUpload(selfie)
        val event = eventRepository.findPublicBySlug(slug)
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
        request: HttpServletRequest,
    ): PaginatedResponse<PhotoDto> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PHOTO_SEARCH, searchKey(principal, request))
        if (principal == null) {
            throw UnauthorizedException(
                code = ErrorCodes.UNAUTHORIZED,
                message = "Sign in to search with a stored selfie.",
            )
        }
        // allSelfies (2026-09-02): match against the whole library, not one
        // pick. Every selfie the runner saved is a different angle of the
        // same face; unioning them is what lets a side-on race photo match.
        if (body.allSelfies == true) {
            val event = eventRepository.findPublicBySlug(slug)
                ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
            val library = userSelfieRepository.findByUserIdOrderByUploadedAtDesc(principal.userId)
            if (library.isEmpty()) {
                throw NotFoundException(code = ErrorCodes.SELFIE_NOT_FOUND, message = "Selfie not found")
            }
            // Skip rows whose object is gone (storage backend changed, deleted
            // out-of-band) rather than failing the whole search on one of them.
            val samples = library.take(MAX_LIBRARY_SELFIES).mapNotNull { s ->
                runCatching { storageService.getBytes(s.s3Key) }.getOrNull()?.let { bytes ->
                    PhotoSearchService.SelfieSample(
                        bytes = bytes,
                        contentType = contentTypeOf(s.s3Key),
                        filename = s.s3Key.substringAfterLast('/'),
                    )
                }
            }
            if (samples.isEmpty()) {
                throw NotFoundException(
                    code = ErrorCodes.SELFIE_NOT_FOUND,
                    message = "Selfie file is no longer available",
                )
            }
            return photoSearchService.searchByFaces(
                eventId = event.id,
                samples = samples,
                pagination = PaginationParams.of(body.offset, body.limit),
                requesterUserId = principal.userId,
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
        val event = eventRepository.findPublicBySlug(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val selfie = userSelfieRepository.findByIdAndUserId(selfieUuid, principal.userId)
            ?: throw NotFoundException(
                code = ErrorCodes.SELFIE_NOT_FOUND,
                message = "Selfie not found",
            )
        // F6 (2026-05-27): StorageService.getBytes throws if the object is
        // missing from S3 (deleted out-of-band, key never written, etc.).
        // Without this wrap the runner gets a generic 500 INTERNAL_ERROR
        // instead of a meaningful 404 they can react to.
        val bytes = runCatching { storageService.getBytes(selfie.s3Key) }.getOrElse {
            throw NotFoundException(
                code = ErrorCodes.SELFIE_NOT_FOUND,
                message = "Selfie file is no longer available",
            )
        }
        return photoSearchService.searchByFace(
            eventId = event.id,
            selfieBytes = bytes,
            contentType = contentTypeOf(selfie.s3Key),
            filename = selfie.s3Key.substringAfterLast('/'),
            pagination = PaginationParams.of(body.offset, body.limit),
            requesterUserId = principal.userId,
        )
    }

    /**
     * The multipart part is unauthenticated, arbitrary bytes on their way to
     * ai-api. Spring's 25 MB multipart ceiling is sized for photographer
     * originals and is far too loose for a selfie. Same whitelist and same
     * 5 MB cap `SelfieService.upload` applies to the stored-selfie path, so
     * both routes into face search agree on what an acceptable selfie is.
     *
     * The stored-selfie JSON path needs no equivalent check — those bytes
     * already cleared this gate when they were uploaded.
     */
    private fun validateSelfieUpload(selfie: MultipartFile) {
        if (selfie.isEmpty) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "selfie file is required",
                field = "selfie",
            )
        }
        val mime = (selfie.contentType ?: "").lowercase().substringBefore(';').trim()
        if (mime !in SUPPORTED_SELFIE_TYPES) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "selfie must be jpeg, png, or webp",
                field = "selfie",
            )
        }
        if (selfie.size > MAX_SELFIE_BYTES) {
            throw ApiException(
                status = HttpStatus.PAYLOAD_TOO_LARGE,
                code = ErrorCodes.PAYLOAD_TOO_LARGE,
                message = "Selfie must be ≤ ${MAX_SELFIE_BYTES / (1024 * 1024)} MB",
                field = "selfie",
            )
        }
    }

    // Bucket key: the signed-in runner when we have one, otherwise the caller's
    // IP — face search is reachable by guests, so a user-only key would leave
    // the anonymous path unthrottled.
    private fun searchKey(principal: AuthPrincipal?, request: HttpServletRequest): String =
        principal?.userId?.toString() ?: clientIp(request)

    private fun contentTypeOf(key: String): String = when (key.substringAfterLast('.').lowercase()) {
        "jpg", "jpeg" -> MediaType.IMAGE_JPEG_VALUE
        "png" -> MediaType.IMAGE_PNG_VALUE
        "webp" -> "image/webp"
        else -> MediaType.APPLICATION_OCTET_STREAM_VALUE
    }

    data class SearchByFaceJsonRequest(
        val selfieId: String? = null,
        // True = ignore selfieId and match with every selfie in the library.
        val allSelfies: Boolean? = null,
        val offset: Int? = null,
        val limit: Int? = null,
    )

    private companion object {
        // Mirrors the selfie-library cap (SELFIE_MAX = 5 on both clients).
        const val MAX_LIBRARY_SELFIES = 5
        const val MAX_SELFIE_BYTES = 5L * 1024 * 1024
        val SUPPORTED_SELFIE_TYPES = setOf("image/jpeg", "image/jpg", "image/png", "image/webp")
    }
}
