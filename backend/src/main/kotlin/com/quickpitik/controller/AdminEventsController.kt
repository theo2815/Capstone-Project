package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminEventDeleteResponseDto
import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.exception.ValidationException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminEventService
import org.springframework.http.MediaType
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@RestController
@RequestMapping("/api/v1/admin/events")
@PreAuthorize("hasRole('ADMIN')")
class AdminEventsController(
    private val adminEventService: AdminEventService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) state: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
        @RequestParam(required = false) q: String?,
        @RequestParam(required = false) dateFrom: String?,
        @RequestParam(required = false) dateTo: String?,
    ): PaginatedResponse<AdminListEventDto> {
        // q + dateFrom/dateTo are accepted for forwards compatibility with
        // the FE param shape but we filter state in-memory; the heavier
        // server-side filters land when the admin volume justifies it.
        @Suppress("UNUSED_VARIABLE") val _q = q
        @Suppress("UNUSED_VARIABLE") val _dateFrom = dateFrom
        @Suppress("UNUSED_VARIABLE") val _dateTo = dateTo
        return adminEventService.list(state, PaginationParams.of(offset, limit))
    }

    // multipart/form-data — text fields ride as @RequestParam (read as
    // plain strings from the multipart parts) and the optional `cover`
    // part carries the image bytes (jpeg/png/webp). @RequestPart on a
    // String would try to use a JSON converter and 500 on a Content-Type-
    // less FormData part. Image bytes belong in object storage, not the
    // JSON payload.
    @PostMapping(consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun create(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam("title") title: String,
        @RequestParam("date") date: String,
        @RequestParam("location") location: String,
        @RequestPart("cover", required = false) cover: MultipartFile?,
    ): AdminListEventDto {
        val req = CreateAdminEventRequest(
            title = title.trim(),
            date = date.trim(),
            location = location.trim(),
            bannerUrl = null,
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
        val coverUpload = cover?.takeUnless { it.isEmpty }?.let {
            AdminEventService.CoverUpload(bytes = it.bytes, contentType = it.contentType)
        }
        return adminEventService.create(principal.userId, req, coverUpload)
    }

    // multipart/form-data so the cover image can ride alongside text fields,
    // mirroring POST. Text-only edits still work — they just send a form
    // with no `cover` part. JSON PATCH was retired 2026-05-16 when the
    // admin-event-form-modal started supporting cover replace/remove.
    @PatchMapping("/{eventId}", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun update(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @RequestParam(required = false) title: String?,
        @RequestParam(required = false) date: String?,
        @RequestParam(required = false) location: String?,
        @RequestParam(required = false) removeCover: Boolean?,
        @RequestPart("cover", required = false) cover: MultipartFile?,
    ): AdminListEventDto {
        val req = UpdateAdminEventRequest(
            title = title?.trim()?.takeIf { it.isNotEmpty() },
            date = date?.trim()?.takeIf { it.isNotEmpty() },
            location = location?.trim()?.takeIf { it.isNotEmpty() },
        )
        val coverUpload = cover?.takeUnless { it.isEmpty }?.let {
            AdminEventService.CoverUpload(bytes = it.bytes, contentType = it.contentType)
        }
        return adminEventService.update(
            adminId = principal.userId,
            eventId = eventId,
            req = req,
            cover = coverUpload,
            removeCover = removeCover == true,
        )
    }

    @DeleteMapping("/{eventId}")
    fun delete(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): AdminEventDeleteResponseDto = adminEventService.delete(principal.userId, eventId)
}
