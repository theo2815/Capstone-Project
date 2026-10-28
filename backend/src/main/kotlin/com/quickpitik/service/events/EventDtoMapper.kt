package com.quickpitik.service.events

import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.dto.events.EventDetailDto
import com.quickpitik.dto.events.EventDto
import com.quickpitik.dto.photos.PhotographerRef
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import java.time.LocalDate
import java.time.ZoneId

/**
 * Single point of truth for Event → DTO conversion.
 *
 * Resolves `bannerUrl` from `cover_s3_key` with a presigned GET (cover TTL)
 * when the admin uploaded a cover via POST /admin/events, and null otherwise
 * (the FE paints a text-only banner fallback). The legacy `events.banner_url`
 * column this used to fall back to was dropped in V25 — it was never written
 * with a non-null value.
 */
@Service
class EventDtoMapper(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val photoRepository: PhotoRepository,
) {
    fun toListDtos(events: List<Event>): List<EventDto> {
        if (events.isEmpty()) return emptyList()
        val counts = photoRepository.countPublicByEventIds(events.map { it.id }).associate { it.eventId to it.photoCount.toInt() }
        return events.map { toListDto(it, counts[it.id] ?: 0) }
    }

    fun toListDto(event: Event, publicPhotoCount: Int = photoRepository.countPublicForEvent(event.id).toInt()): EventDto = EventDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        bannerUrl = resolveBannerUrl(event),
        photoCount = publicPhotoCount,
        participantCount = event.participantCount,
        status = event.status,
        visibility = event.visibility.wire,
        pricingMode = event.pricingMode.wire,
        regionCode = event.regionCode,
        provinceCode = event.provinceCode,
    )

    // `ownerHandle` — the creating photographer's public handle (V46),
    // resolved by the caller; null for admin events.
    fun toDetailDto(event: Event, ownerHandle: String? = null): EventDetailDto = EventDetailDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        bannerUrl = resolveBannerUrl(event),
        photoCount = photoRepository.countPublicForEvent(event.id).toInt(),
        participantCount = event.participantCount,
        status = event.status,
        description = event.description,
        organizerName = event.organizerName,
        categories = event.categories.sorted(),
        pricePerPhoto = event.pricePerPhoto,
        bundlePrice = event.bundlePrice,
        bundleSize = event.bundleSize,
        visibility = event.visibility.wire,
        pricingMode = event.pricingMode.wire,
        watermarkPolicy = event.watermarkPolicy.wire,
        photographerHandle = ownerHandle,
        regionCode = event.regionCode,
        provinceCode = event.provinceCode,
    )

    // `owner` is the photographer who created the event (V46), resolved in
    // batch by the caller; null for admin events.
    fun toAdminListDto(event: Event, owner: PhotographerRef? = null): AdminListEventDto = AdminListEventDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        bannerUrl = resolveBannerUrl(event),
        photoCount = event.photoCount,
        participantCount = event.participantCount,
        status = event.status.name,
        state = deriveAdminEventState(event),
        city = cityFromLocation(event.location),
        pricePerPhoto = event.pricePerPhoto,
        description = event.description,
        organizerName = event.organizerName,
        categories = event.categories.sorted(),
        adminOverrides = event.adminOverrides,
        createdByHandle = owner?.handle,
        createdByName = owner?.name,
        visibility = event.visibility.wire,
        pricingMode = event.pricingMode.wire,
        watermarkPolicy = event.watermarkPolicy.wire,
        reviewStatus = event.reviewStatus.wire,
        reviewNote = event.reviewNote,
        pendingChange = event.pendingChange,
        reviewRevision = event.reviewRevision,
        regionCode = event.regionCode,
        provinceCode = event.provinceCode,
        listedOnEvents = event.listedOnEvents,
    )

    fun resolveBannerUrl(event: Event): String? =
        event.coverS3Key
            ?.takeIf { it.isNotBlank() }
            ?.let { storageService.presignedGetUrl(it, storageProperties.presignedTtl.cover) }

    companion object {
        internal val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")

        // Mirrors website/src/lib/event-catalog.ts. ACTIVE events flip from
        // "upcoming" → "live" on race day (Asia/Manila) and from "live" →
        // "open" once the 4-day upload window closes. The window/archive
        // boundary lives on the FE today; backend only differentiates
        // upcoming / live / open via the date and surfaces archived via
        // EventStatus.ARCHIVED.
        fun deriveAdminEventState(event: Event, today: LocalDate = LocalDate.now(PH_ZONE)): String =
            when (event.status) {
                EventStatus.ACTIVE -> when {
                    event.date.isAfter(today) -> "upcoming"
                    event.date.isEqual(today) -> "live"
                    else -> "open"
                }
                EventStatus.COMPLETED -> "open"
                EventStatus.ARCHIVED -> "past"
                EventStatus.DRAFT -> "upcoming"
            }

        fun cityFromLocation(location: String): String {
            val idx = location.lastIndexOf(", ")
            return if (idx >= 0) location.substring(idx + 2).trim() else location.trim()
        }
    }
}
