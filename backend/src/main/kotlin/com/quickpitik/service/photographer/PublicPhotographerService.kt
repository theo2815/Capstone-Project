package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.CoverSourceDto
import com.quickpitik.dto.photographer.PhotographerEventCoverageDto
import com.quickpitik.dto.photographer.PhotographerProfileDto
import com.quickpitik.dto.photographer.deriveEventState
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.dto.photos.PhotographerRef
import com.quickpitik.dto.photos.toDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.orders.CouponService
import org.springframework.data.domain.PageRequest
import org.springframework.data.domain.Sort
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.format.DateTimeFormatter
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PublicPhotographerService(
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val eventRepository: EventRepository,
    private val photoRepository: PhotoRepository,
    private val storageProperties: StorageProperties,
    private val storageService: com.quickpitik.service.storage.StorageService,
    private val couponService: CouponService,
) {
    fun getProfile(handle: String): PhotographerProfileDto {
        val normalized = handle.trim().lowercase()
        val settings = photographerSettingsRepository.findByHandleIgnoreCase(normalized)
            ?: throw NotFoundException(
                code = ErrorCodes.NOT_FOUND,
                message = "Photographer not found",
            )
        val user = userRepository.findById(settings.userId).orElseThrow {
            // photographer_settings.user_id is FK ON DELETE CASCADE so this is
            // genuinely unreachable, but the !!-free path beats a runtime NPE.
            NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Photographer not found")
        }
        // Suspended or non-approved photographers stay fully invisible to the
        // public — 404 rather than 403 so existence isn't leaked.
        if (user.suspendedAt != null || settings.verificationStatus != VerificationStatus.APPROVED) {
            throw NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Photographer not found")
        }

        // Bounded: this is an UNAUTHENTICATED route and the coverage list was
        // the last unbounded query on one. 200 events ≈ four years of weekly
        // races — a cap, not a pagination contract (wire shape unchanged).
        val coverage = eventPhotographerRepository
            .findAllByIdPhotographerId(
                settings.userId,
                PageRequest.of(0, MAX_PUBLIC_EVENTS, Sort.by(Sort.Direction.DESC, "lastUploadAt")),
            )
            .filter { it.photoCount > 0 }
        val eventsById = if (coverage.isEmpty()) {
            emptyMap()
        } else {
            eventRepository
                .findAllById(coverage.map { it.id.eventId })
                .filter { it.deletedAt == null }
                .associateBy(Event::id)
        }
        val events = coverage.mapNotNull { ep ->
            val event = eventsById[ep.id.eventId] ?: return@mapNotNull null
            PhotographerEventCoverageDto(
                eventSlug = event.slug,
                state = deriveEventState(event),
                photoCount = ep.photoCount,
                salesCount = ep.salesCount,
            )
        }

        val cover = buildCover(settings.coverS3Key, settings.coverGradientFrom, settings.coverGradientTo)

        return PhotographerProfileDto(
            handle = settings.handle ?: normalized,
            displayName = settings.brandName?.takeIf { it.isNotBlank() } ?: user.name,
            brandColor = settings.brandColor,
            bio = settings.bio,
            city = settings.city,
            memberSince = settings.memberSince.format(DateTimeFormatter.ISO_LOCAL_DATE),
            cover = cover,
            watermarkLabel = settings.watermarkLabel,
            events = events,
        )
    }

    fun listEventPhotos(
        handle: String,
        eventSlug: String,
        pagination: PaginationParams,
    ): PaginatedResponse<PhotoDto> {
        val normalized = handle.trim().lowercase()
        val settings = photographerSettingsRepository.findByHandleIgnoreCase(normalized)
            ?: throw NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Photographer not found")
        val user = userRepository.findById(settings.userId).orElseThrow {
            NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Photographer not found")
        }
        if (user.suspendedAt != null || settings.verificationStatus != VerificationStatus.APPROVED) {
            throw NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Photographer not found")
        }
        val event = eventRepository.findBySlugAndDeletedAtIsNull(eventSlug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")

        // Public gallery shows LIVE photos only; HIDDEN / PROCESSING stay
        // invisible. Filter at the query layer so pagination `total` reflects
        // the visible count — a post-fetch filter would leak phantom pages
        // (FE shows N items but the next page is short or empty).
        val page = photoRepository.findPhotographerLibraryByStatus(
            eventId = event.id,
            photographerId = settings.userId,
            status = PhotoStatus.LIVE,
            pageable = OffsetLimitPageable(
                pagination,
                org.springframework.data.domain.Sort
                    .by(org.springframework.data.domain.Sort.Direction.DESC, "uploadedAt")
                    .and(org.springframework.data.domain.Sort.by(org.springframework.data.domain.Sort.Direction.ASC, "id")),
            ),
        )
        if (page.isEmpty) return PaginatedResponse.empty(pagination)
        // Every photo on this page belongs to the photographer resolved above,
        // so attribution is a constant — no per-page lookup like PhotoService
        // needs for a mixed event grid.
        val photographer = PhotographerRef(handle = settings.handle, name = user.name)
        val coupon = couponService.activeFor(setOf(settings.userId))[settings.userId]
        return PaginatedResponse(
            items = page.content.map {
                it.toDto(
                    thumbnailUrlResolver = ::resolveWatermarkedUrl,
                    photographerResolver = { photographer },
                    couponResolver = { photo -> couponService.quoteFor(photo, coupon) },
                )
            },
            total = page.totalElements,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    private fun buildCover(s3Key: String?, gradientFrom: String?, gradientTo: String?): CoverSourceDto? {
        if (s3Key != null) {
            return CoverSourceDto(
                kind = "image",
                url = storageService.presignedGetUrl(s3Key, storageProperties.presignedTtl.cover),
            )
        }
        if (gradientFrom != null && gradientTo != null) {
            return CoverSourceDto(kind = "gradient", from = gradientFrom, to = gradientTo)
        }
        return null
    }

    private fun resolveWatermarkedUrl(photo: Photo): String? {
        // Public gallery serves the watermarked variant when available so the
        // photographer's brand sticks even on incognito browsing. Falls back
        // to the thumbnail (also watermarked in the upload pipeline) and only
        // last to the source key.
        val key = photo.watermarkS3Key ?: photo.thumbnailS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    private companion object {
        // Newest-coverage cap for the unauthenticated profile — ~4 years of
        // weekly races. A bound, not pagination: the wire shape is unchanged.
        const val MAX_PUBLIC_EVENTS = 200
    }
}
