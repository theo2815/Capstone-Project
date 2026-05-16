package com.quickpitik.service.events

import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.dto.events.EventDetailDto
import com.quickpitik.dto.events.EventDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import java.time.LocalDate
import java.time.ZoneId

/**
 * Single point of truth for Event → DTO conversion.
 *
 * Resolves `bannerUrl` from `cover_s3_key` with a presigned GET (cover TTL)
 * when the admin uploaded a cover via POST /admin/events. Falls back to the
 * legacy `events.banner_url` column so V2 seed rows that ship pre-populated
 * URLs continue rendering. The mapper is the only code that knows about the
 * fallback — callers always see a single `bannerUrl` field.
 */
@Service
class EventDtoMapper(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
) {
    fun toListDto(event: Event): EventDto = EventDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        bannerUrl = resolveBannerUrl(event),
        photoCount = event.photoCount,
        participantCount = event.participantCount,
        status = event.status,
    )

    fun toDetailDto(event: Event): EventDetailDto = EventDetailDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        bannerUrl = resolveBannerUrl(event),
        photoCount = event.photoCount,
        participantCount = event.participantCount,
        status = event.status,
        description = event.description,
        organizerName = event.organizerName,
        categories = event.categories.sorted(),
        pricePerPhoto = event.pricePerPhoto,
        bundlePrice = event.bundlePrice,
        bundleSize = event.bundleSize,
    )

    fun toAdminListDto(event: Event): AdminListEventDto = AdminListEventDto(
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
    )

    private fun resolveBannerUrl(event: Event): String? {
        val key = event.coverS3Key
        if (!key.isNullOrBlank()) {
            return storageService.presignedGetUrl(key, storageProperties.presignedTtl.cover)
        }
        return event.bannerUrl
    }

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
