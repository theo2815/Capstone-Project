package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
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
)

// POST /admin/events is now multipart — the controller pulls title/date/
// location/pricePerPhoto off the multipart request and builds this DTO
// before handing to the service. The cover image arrives as a separate
// file part (handled by AdminEventsController + EventCoverService) so the
// row's `cover_s3_key` is set after the event is persisted.
data class CreateAdminEventRequest(
    @field:NotBlank
    @field:Size(max = 200)
    val title: String,
    @field:NotBlank
    val date: String,
    @field:NotBlank
    @field:Size(max = 200)
    val location: String,
    val pricePerPhoto: BigDecimal = BigDecimal.ZERO,
    // Organizer name + race-day notes shown in the runner-facing "About this
    // race" strip. Optional at create — blank leaves the entity defaults ("").
    @field:Size(max = 120)
    val organizerName: String? = null,
    @field:Size(max = 600)
    val description: String? = null,
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
)

data class AdminEventDeleteResponseDto(
    val removed: Boolean,
)

data class AdminReindexResponseDto(
    val requeued: Int,
)
