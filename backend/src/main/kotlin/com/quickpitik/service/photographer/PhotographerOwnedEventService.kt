package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.CreateMyEventRequest
import com.quickpitik.dto.photographer.PhotographerEventDetailDto
import com.quickpitik.dto.photographer.UpdateMyEventRequest
import com.quickpitik.dto.photographer.detailDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.events.EventInputs
import com.quickpitik.websocket.AdminInboxEvent
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.OffsetDateTime
import java.util.UUID

// The pricing trio a photographer chooses for their own event. Parsed and
// validated in one place: FREE ⇒ price 0 and OWN|NONE (default OWN); PAID ⇒
// price > 0 and always the PLATFORM mark.
data class PricingTrio(
    val pricingMode: EventPricingMode,
    val pricePerPhoto: BigDecimal,
    val watermarkPolicy: WatermarkPolicy,
) {
    fun sameAs(other: PricingTrio): Boolean =
        pricingMode == other.pricingMode &&
            watermarkPolicy == other.watermarkPolicy &&
            pricePerPhoto.compareTo(other.pricePerPhoto) == 0

    fun applyTo(event: Event) {
        event.pricingMode = pricingMode
        event.pricePerPhoto = pricePerPhoto
        event.watermarkPolicy = watermarkPolicy
    }

    fun toJson(now: OffsetDateTime = OffsetDateTime.now()): Map<String, Any?> = mapOf(
        "pricingMode" to pricingMode.wire,
        "pricePerPhoto" to pricePerPhoto.toPlainString(),
        "watermarkPolicy" to watermarkPolicy.wire,
        "requestedAt" to now.toString(),
    )

    companion object {
        fun of(event: Event) = PricingTrio(event.pricingMode, event.pricePerPhoto, event.watermarkPolicy)

        fun fromJson(json: Map<String, Any?>): PricingTrio = parse(
            json["pricingMode"] as? String,
            (json["pricePerPhoto"] as? String)?.let { BigDecimal(it) },
            json["watermarkPolicy"] as? String,
        )

        fun parse(modeWire: String?, price: BigDecimal?, policyWire: String?): PricingTrio {
            val mode = EventPricingMode.fromWire(modeWire ?: EventPricingMode.PAID.wire)
                ?: throw ValidationException(
                    message = "pricingMode must be paid or free",
                    code = ErrorCodes.VALIDATION_ERROR,
                    field = "pricingMode",
                )
            return when (mode) {
                EventPricingMode.FREE -> {
                    val policy = policyWire?.takeIf { it.isNotBlank() }?.let {
                        WatermarkPolicy.fromWire(it) ?: throw ValidationException(
                            message = "watermarkPolicy must be own or none",
                            code = ErrorCodes.VALIDATION_ERROR,
                            field = "watermarkPolicy",
                        )
                    } ?: WatermarkPolicy.OWN
                    if (policy == WatermarkPolicy.PLATFORM) {
                        throw ValidationException(
                            message = "A free event carries your own logo or no watermark — never the QuickPitik mark",
                            code = ErrorCodes.VALIDATION_ERROR,
                            field = "watermarkPolicy",
                        )
                    }
                    PricingTrio(EventPricingMode.FREE, BigDecimal.ZERO.setScale(2), policy)
                }
                EventPricingMode.PAID -> {
                    val p = price ?: throw ValidationException(
                        message = "pricePerPhoto is required for a paid event",
                        code = ErrorCodes.VALIDATION_ERROR,
                        field = "pricePerPhoto",
                    )
                    if (p.signum() <= 0) {
                        throw ValidationException(
                            message = "pricePerPhoto must be greater than 0",
                            code = ErrorCodes.VALIDATION_ERROR,
                            field = "pricePerPhoto",
                        )
                    }
                    PricingTrio(EventPricingMode.PAID, p.setScale(2, RoundingMode.HALF_UP), WatermarkPolicy.PLATFORM)
                }
            }
        }
    }
}

// Photographer-owned events (V46). A photographer creates an event, an admin
// approves it (AdminEventService.approve) before uploads open, and once live
// the pricing trio can only change through a request the admin applies —
// this service never writes the live trio itself.
@Service
@Transactional
class PhotographerOwnedEventService(
    private val eventRepository: EventRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val eventCoverService: EventCoverService,
    private val eventDtoMapper: EventDtoMapper,
    private val eventPublisher: ApplicationEventPublisher,
) {
    fun create(
        photographerId: UUID,
        req: CreateMyEventRequest,
        cover: AdminEventService.CoverUpload?,
    ): PhotographerEventDetailDto {
        requireApprovedPhotographer(photographerId)
        val trio = PricingTrio.parse(req.pricingMode, req.pricePerPhoto, req.watermarkPolicy)
        val visibility = parseVisibility(req.visibility)
        val now = OffsetDateTime.now()
        val event = eventRepository.save(
            Event(
                slug = EventInputs.slugify(req.title),
                name = req.title.trim(),
                date = EventInputs.parseDate(req.date),
                location = req.location.trim(),
                status = EventStatus.DRAFT,
                description = req.description?.trim().orEmpty(),
                organizerName = req.organizerName?.trim().orEmpty(),
                pricePerPhoto = trio.pricePerPhoto,
                createdAt = now,
                updatedAt = now,
                createdBy = photographerId,
                visibility = visibility,
                pricingMode = trio.pricingMode,
                watermarkPolicy = trio.watermarkPolicy,
                reviewStatus = EventReviewStatus.PENDING,
            ),
        )
        if (cover != null) {
            event.coverS3Key = eventCoverService.upload(event.id, cover.bytes, cover.contentType)
            eventRepository.save(event)
        }
        // Coverage row up front so the event shows in /me/photographer/events
        // before the first upload; the upload upsert bumps its counters later.
        val ep = eventPhotographerRepository.save(
            EventPhotographer(id = EventPhotographerId(event.id, photographerId)),
        )
        notifyAdmins("event_submitted", event, photographerId)
        return detailDto(event, ep, eventDtoMapper.resolveBannerUrl(event), viewerId = photographerId)
    }

    fun update(
        photographerId: UUID,
        eventId: UUID,
        req: UpdateMyEventRequest,
        cover: AdminEventService.CoverUpload?,
    ): PhotographerEventDetailDto {
        val event = eventRepository.findByIdAndCreatedByAndDeletedAtIsNull(eventId, photographerId)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")

        // Presentation + visibility apply straight away in every state.
        req.title?.takeIf { it.isNotBlank() }?.let { event.name = it.trim() }
        req.date?.takeIf { it.isNotBlank() }?.let { event.date = EventInputs.parseDate(it) }
        req.location?.takeIf { it.isNotBlank() }?.let { event.location = it.trim() }
        req.organizerName?.takeIf { it.isNotBlank() }?.let { event.organizerName = it.trim() }
        req.description?.takeIf { it.isNotBlank() }?.let { event.description = it.trim() }
        req.visibility?.takeIf { it.isNotBlank() }?.let { event.visibility = parseVisibility(it) }
        if (cover != null) {
            val old = event.coverS3Key
            event.coverS3Key = eventCoverService.upload(event.id, cover.bytes, cover.contentType)
            if (!old.isNullOrBlank()) eventCoverService.delete(old)
        }

        if (req.withdrawPendingChange && event.reviewStatus == EventReviewStatus.CHANGE_PENDING) {
            event.pendingChange = null
            event.reviewStatus = EventReviewStatus.APPROVED
        }

        val requested = requestedTrio(event, req)
        when (event.reviewStatus) {
            // Not live yet: the trio is still the photographer's to set.
            EventReviewStatus.PENDING, EventReviewStatus.REJECTED -> {
                requested?.applyTo(event)
                if (event.reviewStatus == EventReviewStatus.REJECTED) {
                    event.reviewStatus = EventReviewStatus.PENDING
                    event.reviewNote = null
                    notifyAdmins("event_submitted", event, photographerId)
                }
            }
            // Live: the trio is frozen. A differing request is parked for the
            // admin; the event keeps selling on its current settings.
            EventReviewStatus.APPROVED, EventReviewStatus.CHANGE_PENDING -> {
                if (requested != null && !requested.sameAs(PricingTrio.of(event))) {
                    event.pendingChange = requested.toJson()
                    event.reviewStatus = EventReviewStatus.CHANGE_PENDING
                    notifyAdmins("event_change_requested", event, photographerId)
                }
            }
        }
        eventRepository.save(event)
        val ep = eventPhotographerRepository.findById(EventPhotographerId(event.id, photographerId))
            .orElseGet { EventPhotographer(id = EventPhotographerId(event.id, photographerId)) }
        return detailDto(event, ep, eventDtoMapper.resolveBannerUrl(event), viewerId = photographerId)
    }

    // A request that names none of the three fields is not a pricing edit.
    // Missing fields inherit from the event, except that a FREE request with
    // no policy defaults to OWN (PLATFORM is never valid for free).
    private fun requestedTrio(event: Event, req: UpdateMyEventRequest): PricingTrio? {
        if (req.pricingMode == null && req.pricePerPhoto == null && req.watermarkPolicy == null) return null
        val policy = req.watermarkPolicy
            ?: event.watermarkPolicy.takeIf { it != WatermarkPolicy.PLATFORM }?.wire
        return PricingTrio.parse(
            req.pricingMode ?: event.pricingMode.wire,
            req.pricePerPhoto ?: event.pricePerPhoto.takeIf { it.signum() > 0 },
            policy,
        )
    }

    private fun parseVisibility(raw: String): EventVisibility =
        EventVisibility.fromWire(raw) ?: throw ValidationException(
            message = "visibility must be public or unlisted",
            code = ErrorCodes.VALIDATION_ERROR,
            field = "visibility",
        )

    // Same gate as uploads: a suspended or unverified photographer creates
    // nothing (PhotoUploadService.gate keeps the same codes).
    private fun requireApprovedPhotographer(photographerId: UUID) {
        val user = userRepository.findById(photographerId).orElse(null)
            ?: throw NotFoundException(code = ErrorCodes.USER_NOT_FOUND, message = "User not found")
        if (user.suspendedAt != null) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.ACCOUNT_SUSPENDED,
                message = "Your account is suspended. Contact support to appeal before creating events.",
            )
        }
        val settings = photographerSettingsRepository.findById(photographerId).orElse(null)
        if (settings == null || settings.verificationStatus != VerificationStatus.APPROVED) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED,
                message = "Submit your photographer verification before creating an event.",
            )
        }
    }

    // Pings connected admins so the event queue refreshes without a reload —
    // same AFTER_COMMIT publish PhotographerSettingsService.submitVerification uses.
    private fun notifyAdmins(type: String, event: Event, photographerId: UUID) {
        eventPublisher.publishEvent(
            AdminInboxEvent(
                payload = mapOf(
                    "type" to type,
                    "entityId" to event.id.toString(),
                    "actorId" to photographerId.toString(),
                    "occurredAt" to OffsetDateTime.now().toString(),
                ),
            ),
        )
    }
}
