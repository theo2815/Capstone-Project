package com.quickpitik.service.cart

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.cart.CartItemDto
import com.quickpitik.dto.cart.MergeCartItem
import com.quickpitik.entity.CartItemEntity
import com.quickpitik.entity.CartItemId
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.CartItemRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.photos.PhotographerSuspension
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.RoundingMode
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

@Service
@Transactional
class CartService(
    private val cartItemRepository: CartItemRepository,
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val photographerSuspension: PhotographerSuspension,
) {
    /**
     * A suspended photographer's photo drops out of the list (its row is kept, so it comes back on
     * unsuspend), or shows with `available = false` when the client asks (AA-I12b, C13).
     */
    @Transactional(readOnly = true)
    fun list(userId: UUID, includeUnavailable: Boolean = false): List<CartItemDto> {
        val rows = cartItemRepository.findByUserId(userId)
        if (rows.isEmpty()) return emptyList()
        return hydrate(rows, includeUnavailable)
    }

    fun add(userId: UUID, photoId: UUID, eventId: UUID): CartItemDto {
        val photo = photoRepository.findById(photoId).orElseThrow {
            NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        }
        // PROCESSING / HIDDEN photos are not sellable. 404 rather than a
        // dedicated code — a hidden photo must not be distinguishable from a
        // missing one.
        if (!photo.isSellable || photographerSuspension.suspendedOwnerIds(listOf(photo)).isNotEmpty()) {
            throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        }
        // Free events (V46): a ₱0 photo is downloaded from the gallery, never
        // bought — OrderService would refuse it at checkout anyway.
        if (photo.pricePhp.signum() <= 0) {
            throw ValidationException(
                message = "This photo is free — download it from the gallery",
                code = ErrorCodes.PHOTO_FREE,
                field = "photoId",
            )
        }
        if (photo.eventId != eventId) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Photo does not belong to that event",
                field = "eventId",
            )
        }
        val event = eventRepository.findById(eventId).orElseThrow {
            NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
        if (event.status == EventStatus.ARCHIVED) {
            throw ConflictException(code = ErrorCodes.EVENT_ARCHIVED, message = "Event is archived")
        }
        val id = CartItemId(userId = userId, photoId = photoId)
        val existing = cartItemRepository.findById(id).orElse(null)
        if (existing != null && existing.pricePhpAtAdd.compareTo(photo.pricePhp) != 0) {
            throw ConflictException(
                code = ErrorCodes.CART_ITEM_PRICE_CHANGED,
                message = "Photo price changed from ₱${existing.pricePhpAtAdd} to ₱${photo.pricePhp}.",
            )
        }
        val saved = if (existing != null) {
            existing.eventId = eventId
            cartItemRepository.save(existing)
        } else {
            cartItemRepository.save(
                CartItemEntity(
                    id = id,
                    eventId = eventId,
                    pricePhpAtAdd = photo.pricePhp,
                ),
            )
        }
        return hydrate(listOf(saved)).first()
    }

    fun remove(userId: UUID, photoId: UUID): Boolean =
        cartItemRepository.deleteByUserIdAndPhotoId(userId, photoId) > 0

    fun clear(userId: UUID): Int = cartItemRepository.deleteAllByUserId(userId)

    /**
     * Guest → authed cart merge, fired once per login.
     *
     * Every gate below **skips** the offending row where [add] **throws**. The
     * asymmetry is deliberate: [add] is a single-item interactive call where a
     * 409 is actionable, while merge is a bulk background call the FE fires
     * inside a `Promise.all` with no partial-recovery path — one rejection
     * discards the whole merge (cart *and* saved-events), and price drift never
     * self-heals, so every later login would re-fail and strand the guest cart.
     *
     * Same reason price drift never raises CART_ITEM_PRICE_CHANGED here. The
     * `pricePhpAtAdd` snapshot is now displayed (AA-R14: "went up from ₱X")
     * and lives as long as the row, so merge never overwrites an existing
     * row's snapshot. A new row takes the guest's `priceAtAdd` only when
     * 0 < value ≤ the live price (a client cannot claim a higher one), else
     * the live price. What the runner pays is still the live price.
     */
    fun merge(userId: UUID, incoming: List<MergeCartItem>): List<CartItemDto> {
        val existing = cartItemRepository.findByUserId(userId)
            .associateBy { it.id.photoId }
            .toMutableMap()
        val incomingIds = incoming.map { it.photoId }.toSet()
        val photoLookup = photoRepository.findAllById(incomingIds).associateBy { it.id }
        val eventLookup = eventRepository.findAllById(incoming.map { it.eventId }.toSet())
            .associateBy { it.id }
        val suspended = photographerSuspension.suspendedOwnerIds(photoLookup.values)
        for ((photoId, eventId, guestPriceAtAdd) in incoming) {
            val photo = photoLookup[photoId] ?: continue
            if (photo.eventId != eventId) continue
            if (!photo.isSellable || photo.photographerId in suspended) continue
            // A free photo (V46) would 409 every checkout after login — skip.
            if (photo.pricePhp.signum() <= 0) continue
            val event = eventLookup[eventId] ?: continue
            if (event.status == EventStatus.ARCHIVED) continue
            val key = CartItemId(userId, photoId)
            val row = existing[photoId]
            if (row == null) {
                val guest = guestPriceAtAdd?.setScale(2, RoundingMode.HALF_UP)
                val saved = cartItemRepository.save(
                    CartItemEntity(
                        id = key,
                        eventId = eventId,
                        pricePhpAtAdd = guest?.takeIf { it.signum() > 0 && it <= photo.pricePhp } ?: photo.pricePhp,
                    ),
                )
                existing[photoId] = saved
            } else {
                row.eventId = eventId
                cartItemRepository.save(row)
            }
        }
        return list(userId)
    }

    private fun hydrate(rows: List<CartItemEntity>, includeUnavailable: Boolean = false): List<CartItemDto> {
        if (rows.isEmpty()) return emptyList()
        val photoIds = rows.map { it.id.photoId }.toSet()
        val eventIds = rows.map { it.eventId }.toSet()
        val photos = photoRepository.findAllById(photoIds).associateBy { it.id }
        val events = eventRepository.findAllById(eventIds).associateBy { it.id }
        val suspended = photographerSuspension.suspendedOwnerIds(photos.values.filter { it.isSellable })
        return rows.mapNotNull { row ->
            val photo = photos[row.id.photoId] ?: return@mapNotNull null
            if (!photo.isSellable) return@mapNotNull null
            val available = photo.photographerId !in suspended
            if (!available && !includeUnavailable) return@mapNotNull null
            val event = events[row.eventId]
            row.toDto(photo, event, available)
        }
    }

    // `price` is the LIVE `photos.price_php`: `OrderService.create` totals and
    // charges it. `priceAtAdd` is the snapshot, shown only as "went up from
    // ₱priceAtAdd" while `price` is higher (AA-R14), never as the price.
    private fun CartItemEntity.toDto(photo: Photo, event: Event?, available: Boolean): CartItemDto = CartItemDto(
        photoId = id.photoId,
        eventId = eventId,
        thumbnailUrl = thumbnailUrlOf(photo),
        price = photo.pricePhp,
        bib = photo.bibs.minByOrNull { it.bibNumber }?.bibNumber,
        eventName = event?.name,
        eventSlug = event?.slug,
        tone = photo.tone,
        time = (photo.capturedAt ?: photo.uploadedAt)
            .atZoneSameInstant(DISPLAY_ZONE)
            .toLocalTime()
            .format(TIME_FORMATTER),
        available = available,
        priceAtAdd = pricePhpAtAdd,
    )

    private fun thumbnailUrlOf(photo: Photo): String {
        val key = photo.watermarkS3Key ?: photo.thumbnailS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    private companion object {
        val TIME_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm")
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
    }
}
