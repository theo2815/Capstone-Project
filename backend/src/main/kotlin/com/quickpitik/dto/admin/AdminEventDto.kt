package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import jakarta.validation.constraints.PositiveOrZero
import io.swagger.v3.oas.annotations.media.Schema
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID

// Mirrors website/src/app/events/events-browser.tsx ListEvent =
// Event & { state, city }. Admin uses the same shape so the /admin/events
// surface can drop into the existing event-tile renderer.
//
// `adminOverrides` exposes the per-row override history persisted by
// AdminEventService.update (Q-A3 / C-5). Each entry shape is
// `{ at, adminId, before, after }`; the FE can render a history affordance
// without joining the cross-event admin_decision_log surface. Defaults
// empty so a new event from POST /admin/events ships with `[]`.
data class AdminListEventDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val location: String,
    val bannerUrl: String?,
    val photoCount: Int,
    val participantCount: Int,
    val status: String,
    val state: String,
    val city: String,
    val pricePerPhoto: BigDecimal,
    val description: String,
    val organizerName: String,
    val categories: List<String>,
    val adminOverrides: List<Map<String, Any?>> = emptyList(),
    // Photographer-owned events (V46). createdBy* null = admin event.
    val createdByHandle: String? = null,
    val createdByName: String? = null,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val reviewStatus: String = "approved",
    val reviewNote: String? = null,
    val pendingChange: Map<String, Any?>? = null,
    @field:Schema(description = "Event review generation; advances on actual decisions and edits")
    val reviewRevision: Long = 0,
    @field:Schema(description = "Canonical region code from /regions; null for unclassified legacy events", nullable = true)
    val regionCode: String? = null,
    @field:Schema(description = "Canonical province code from /regions; null for unclassified legacy events", nullable = true)
    val provinceCode: String? = null,
    @field:Schema(description = "Admin's \"Show on /events\" switch for a photographer-owned Public event (V89)")
    val listedOnEvents: Boolean = false,
)

// PATCH /admin/events/{id}/listing — show or hide a photographer event on /events.
data class SetEventListingRequest(
    val listed: Boolean,
)

data class ApproveEventRequest(
    @field:PositiveOrZero
    @field:Schema(description = "Optional v1 precondition; stale generations return 409 before side effects")
    val expectedRevision: Long? = null,
)

// POST /admin/events/{id}/reject — the reason reaches the photographer's inbox.
data class RejectEventRequest(
    @field:NotBlank
    @field:Size(max = 500)
    val reason: String,
    @field:PositiveOrZero
    val expectedRevision: Long? = null,
)

// POST /admin/events is multipart — the text parts bind straight onto this
// DTO as a `@Valid @ModelAttribute` (AdminEventsController). The cover image
// arrives as a separate file part (handled by AdminEventsController +
// EventCoverService) before the row and its `cover_s3_key` are persisted.
// The three required fields carry an empty default so a missing
// part fails @NotBlank as a 400 on that field rather than a construction error.
data class CreateAdminEventRequest(
    @field:NotBlank
    @field:Size(max = 200)
    val title: String = "",
    @field:NotBlank
    val date: String = "",
    @field:NotBlank
    @field:Size(max = 200)
    val location: String = "",
    val pricePerPhoto: BigDecimal = BigDecimal.ZERO,
    // Organizer name + race-day notes shown in the runner-facing "About this
    // race" strip. Optional at create — blank leaves the entity defaults ("").
    @field:Size(max = 120)
    val organizerName: String? = null,
    @field:Size(max = 600)
    val description: String? = null,
    @field:Size(max = 20)
    @field:Schema(description = "Optional v1 location pair from /regions; supply both nonblank codes or omit both")
    val regionCode: String? = null,
    @field:Size(max = 40)
    val provinceCode: String? = null,
)

// PATCH /admin/events/{id} — body { title?, date?, location?, pricePerPhoto?,
// organizerName?, description? }. Admin-editable fields; status and slug stay
// fixed (slug to keep public URLs stable, status to keep state machines
// out of the admin surface). When `pricePerPhoto` changes, AdminEventService
// also re-prices existing `photos.price_php` rows under the same event so the
// new price takes effect across runner-facing galleries and carts.
// organizerName + description feed the runner-facing "About this race" strip.
data class UpdateAdminEventRequest(
    val title: String? = null,
    val date: String? = null,
    val location: String? = null,
    val pricePerPhoto: BigDecimal? = null,
    @field:Size(max = 120)
    val organizerName: String? = null,
    @field:Size(max = 600)
    val description: String? = null,
    @field:PositiveOrZero
    val expectedRevision: Long? = null,
    @field:Size(max = 20)
    @field:Schema(description = "Omit both codes to preserve location; supply both valid /regions codes to replace it. Clearing is not supported")
    val regionCode: String? = null,
    @field:Size(max = 40)
    val provinceCode: String? = null,
)

data class AdminEventDeleteResponseDto(
    val removed: Boolean,
)

data class AdminReindexResponseDto(
    val requeued: Int,
)
