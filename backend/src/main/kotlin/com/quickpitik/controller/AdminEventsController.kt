package com.quickpitik.controller

import com.quickpitik.common.IdempotencyKey
import org.springframework.web.bind.annotation.RequestHeader
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.ApproveEventRequest
import com.quickpitik.dto.admin.AdminEventDeleteResponseDto
import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.dto.admin.AdminReindexResponseDto
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.admin.RejectEventRequest
import com.quickpitik.dto.admin.SetEventListingRequest
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminEventService
import jakarta.validation.Valid
import org.springframework.beans.propertyeditors.StringTrimmerEditor
import org.springframework.http.MediaType
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.WebDataBinder
import org.springframework.web.bind.annotation.DeleteMapping
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
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@RestController
@RequestMapping("/api/v1/admin/events")
@PreAuthorize("hasRole('ADMIN')")
class AdminEventsController(
    private val adminEventService: AdminEventService,
) {

    // `review=queue` (V46) lists photographer-owned events awaiting a
    // decision instead of the full pipeline.
    @GetMapping
    fun list(
        @RequestParam(required = false) state: String?,
        @RequestParam(required = false) review: String?,
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
        return adminEventService.list(state, review, PaginationParams.of(offset, limit))
    }

    // ── Photographer-owned event review (V46) ────────────────────────────
    @PostMapping("/{eventId}/approve")
    fun approve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody(required = false) body: ApproveEventRequest? = null,
    ): AdminListEventDto = adminEventService.approve(principal.userId, eventId, body?.expectedRevision)

    @PostMapping("/{eventId}/reject")
    fun reject(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: RejectEventRequest,
    ): AdminListEventDto = adminEventService.reject(principal.userId, eventId, body.reason, body.expectedRevision)

    // "Show on /events" switch for a photographer-owned Public event (V89).
    @PatchMapping("/{eventId}/listing")
    fun setListing(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @RequestBody body: SetEventListingRequest,
    ): AdminListEventDto = adminEventService.setListing(principal.userId, eventId, body.listed)

    // multipart/form-data — the text parts bind onto the request DTO and the
    // optional `cover` part carries the image bytes (jpeg/png/webp).
    // @RequestPart on a String would try to use a JSON converter and 500 on a
    // Content-Type-less FormData part. Image bytes belong in object storage,
    // not the JSON payload.
    @PostMapping(consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun create(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @ModelAttribute("form") form: CreateAdminEventRequest,
        @RequestPart("cover", required = false) cover: MultipartFile?,
        @RequestHeader(IdempotencyKey.HEADER, required = false) key: String? = null,
    ): AdminListEventDto = adminEventService.create(principal.userId, form, coverOf(cover), IdempotencyKey.parse(key))

    // multipart/form-data so the cover image can ride alongside text fields,
    // mirroring POST. Text-only edits still work — they just send a form
    // with no `cover` part. JSON PATCH was retired 2026-05-16 when the
    // admin-event-form-modal started supporting cover replace/remove.
    @PatchMapping("/{eventId}", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun update(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @ModelAttribute("form") form: UpdateAdminEventRequest,
        @RequestParam(required = false) removeCover: Boolean?,
        @RequestPart("cover", required = false) cover: MultipartFile?,
        @RequestHeader(IdempotencyKey.HEADER, required = false) key: String? = null,
    ): AdminListEventDto = adminEventService.update(
        adminId = principal.userId,
        eventId = eventId,
        req = form,
        cover = coverOf(cover),
        removeCover = removeCover == true,
        key = IdempotencyKey.parse(key),
    )

    @DeleteMapping("/{eventId}")
    fun delete(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): AdminEventDeleteResponseDto = adminEventService.delete(principal.userId, eventId)

    // Re-drive AI indexing for this event's photos. Default requeues
    // FAILED/PARTIAL (outage recovery); ?all=true also requeues INDEXED +
    // SKIPPED (provider flip, or AI enabled after the photos were uploaded).
    @PostMapping("/{eventId}/photos/reindex")
    fun reindexPhotos(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @RequestParam(required = false) all: Boolean?,
    ): AdminReindexResponseDto =
        AdminReindexResponseDto(requeued = adminEventService.reindexPhotos(principal.userId, eventId, all == true))

    // FormData sends every field as a string and an untouched input as "", so
    // blank has to read as "absent": "default to 0" on POST, "no change" on
    // PATCH. Scoped by the `form` attribute name to the two event forms.
    // Numbers convert themselves, and a non-numeric price comes back as a 400
    // on that field so the FE shows it on the right input, not a generic 500.
    @InitBinder("form")
    fun trimEventForm(binder: WebDataBinder) {
        binder.registerCustomEditor(String::class.java, StringTrimmerEditor(true))
        // A supplied blank pair is invalid, not an omitted PATCH.
        binder.registerCustomEditor(String::class.java, "regionCode", StringTrimmerEditor(false))
        binder.registerCustomEditor(String::class.java, "provinceCode", StringTrimmerEditor(false))
    }

    private fun coverOf(cover: MultipartFile?): AdminEventService.CoverUpload? =
        cover?.takeUnless { it.isEmpty }?.let {
            AdminEventService.CoverUpload(bytes = it.bytes, contentType = it.contentType)
        }
}
