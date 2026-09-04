package com.quickpitik.service.photos

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.dto.photos.PhotographerRef
import com.quickpitik.dto.photos.toDto
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.orders.CouponService
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PhotoService(
    private val photoRepository: PhotoRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val couponService: CouponService,
    private val eventRepository: EventRepository,
) {
    fun listForEvent(
        eventId: UUID,
        bib: String?,
        pagination: PaginationParams,
        requesterUserId: UUID? = null,
        snapshotAt: OffsetDateTime? = null,
    ): PaginatedResponse<PhotoDto> {
        val cleanedBib = normalizeBib(bib)
        // Plain browsing (no bib) takes the join-free fast path — the grid is
        // the hottest read in the system and needs neither the bibs join nor
        // the DISTINCT it forces.
        val gallerySnapshot = if (cleanedBib.isEmpty()) {
            snapshotAt ?: OffsetDateTime.now(ZoneOffset.UTC).withNano(0)
        } else {
            null
        }
        val page = if (gallerySnapshot != null) {
            photoRepository.findForEventNoBib(
                eventId = eventId,
                snapshotAt = gallerySnapshot,
                seed = gallerySnapshot.toEpochSecond(),
                pageable = OffsetLimitPageable(pagination),
            )
        } else {
            photoRepository.searchForEvent(
                eventId = eventId,
                status = PhotoStatus.LIVE,
                bib = cleanedBib,
                pageable = OffsetLimitPageable(pagination),
            )
        }
        return toPaginatedDto(page.content, page.totalElements, pagination, requesterUserId, eventId)
            .copy(snapshotAt = gallerySnapshot)
    }

    fun findByEventAndPersonIds(
        eventId: UUID,
        aiPersonIds: Collection<String>,
        pagination: PaginationParams,
        requesterUserId: UUID? = null,
    ): PaginatedResponse<PhotoDto> {
        if (aiPersonIds.isEmpty()) {
            return PaginatedResponse.empty(pagination)
        }
        val page = photoRepository.findByEventAndPersonIds(
            eventId = eventId,
            status = PhotoStatus.LIVE,
            aiPersonIds = aiPersonIds,
            pageable = OffsetLimitPageable(pagination),
        )
        return toPaginatedDto(page.content, page.totalElements, pagination, requesterUserId, eventId)
    }

    private fun toPaginatedDto(
        photos: List<Photo>,
        total: Long,
        pagination: PaginationParams,
        requesterUserId: UUID?,
        eventId: UUID,
    ): PaginatedResponse<PhotoDto> {
        // Free event (V46): the original is anyone's, so no grant lookup and
        // every visitor gets the clean + download URLs.
        val free = eventRepository.findById(eventId).map { it.isFree }.orElse(false)
        val ownedIds = if (free) emptySet() else resolveOwnedIds(requesterUserId, photos)
        val photographers = resolvePhotographers(photos)
        // Live coupons for the page's photographers — one IN query, same
        // batch shape as attribution.
        val coupons = couponService.activeFor(eventId, photographers.keys)
        return PaginatedResponse(
            items = photos.map {
                it.toDto(
                    thumbnailUrlResolver = ::resolveThumbnailUrl,
                    cleanUrlResolver = { photo ->
                        if (free || photo.id in ownedIds) resolveCleanUrl(photo) else null
                    },
                    photographerResolver = { photo ->
                        photo.photographerId?.let { photographers[it] }
                    },
                    couponResolver = { photo ->
                        couponService.quoteFor(photo, photo.photographerId?.let { coupons[it] })
                    },
                    downloadUrlResolver = { photo -> if (free) resolveDownloadUrl(photo) else null },
                    free = free,
                )
            },
            total = total,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    private fun resolveOwnedIds(requesterUserId: UUID?, photos: List<Photo>): Set<UUID> {
        if (requesterUserId == null || photos.isEmpty()) return emptySet()
        return downloadGrantRepository
            .findOwnedPhotoIdsByUserAndPhotoIds(requesterUserId, photos.map { it.id })
            .toSet()
    }

    // Photographer attribution for a whole page in two reads, not two per photo.
    // A marathon event page is 60-240 photos from a handful of photographers, so
    // the distinct-id set is tiny and findAllById collapses to one IN query each.
    // Mirrors resolveOwnedIds above: batch here, map lookup in the resolver.
    //
    // handle comes from PhotographerSettings (null until verification assigns
    // one), name from User. A photographer with settings but no handle still
    // gets a name, which is why the two are looked up independently rather than
    // skipping the user read when the handle is missing.
    private fun resolvePhotographers(photos: List<Photo>): Map<UUID, PhotographerRef> {
        if (photos.isEmpty()) return emptyMap()
        // photographerId is nullable on Photo (legacy/seed rows predate the
        // column), so drop the nulls here — those photos simply carry no
        // attribution and the resolver returns null for them.
        val photographerIds = photos.mapNotNullTo(mutableSetOf()) { it.photographerId }
        if (photographerIds.isEmpty()) return emptyMap()
        val handles = photographerSettingsRepository.findAllById(photographerIds)
            .associate { it.userId to it.handle }
        val names = userRepository.findAllById(photographerIds)
            .associate { it.id to it.name }
        return photographerIds.associateWith { id ->
            PhotographerRef(handle = handles[id], name = names[id])
        }
    }

    private fun resolveThumbnailUrl(photo: Photo): String? {
        val key = photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    // The clean original. Reuses the runner-download TTL since the lightbox
    // surface and the actual download share the same security envelope —
    // both are gated on proof of ownership upstream.
    private fun resolveCleanUrl(photo: Photo): String =
        storageService.presignedGetUrl(photo.s3Key, storageProperties.presignedTtl.runnerDownload)

    // Attachment-disposition URL for a free photo's original, same TTL and
    // filename rule as the order download.
    private fun resolveDownloadUrl(photo: Photo): String =
        storageService.presignedDownloadUrl(
            photo.s3Key,
            storageProperties.presignedTtl.runnerDownload,
            PhotoFilenames.downloadFilenameOf(photo),
        )

    private fun normalizeBib(raw: String?): String {
        if (raw.isNullOrBlank()) return ""
        return raw.removePrefix("B-").removePrefix("b-").trim().uppercase()
    }
}
