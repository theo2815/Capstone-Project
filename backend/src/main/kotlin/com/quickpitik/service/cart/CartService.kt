package com.quickpitik.service.cart

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.cart.CartItemDto
import com.quickpitik.entity.CartItemEntity
import com.quickpitik.entity.CartItemId
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.CartItemRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
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
) {
    @Transactional(readOnly = true)
    fun list(userId: UUID): List<CartItemDto> {
        val rows = cartItemRepository.findByUserId(userId)
        if (rows.isEmpty()) return emptyList()
        return hydrate(rows)
    }

    fun add(userId: UUID, photoId: UUID, eventId: UUID): CartItemDto {
        val photo = photoRepository.findById(photoId).orElseThrow {
            NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        }
        // PROCESSING / HIDDEN photos are not sellable. 404 rather than a
        // dedicated code — a hidden photo must not be distinguishable from a
        // missing one.
        if (photo.status != PhotoStatus.LIVE) {
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
     * Same reason `pricePhpAtAdd` is refreshed rather than raising
     * CART_ITEM_PRICE_CHANGED here: the snapshot is not what the runner sees
     * or pays. [toDto] renders `photos.price_php` and `OrderService.create`
     * charges it, so refreshing the column keeps the stored row honest
     * instead of preserving a number nothing reads.
     */
    fun merge(userId: UUID, incoming: List<Pair<UUID, UUID>>): List<CartItemDto> {
        val existing = cartItemRepository.findByUserId(userId)
            .associateBy { it.id.photoId }
            .toMutableMap()
        val incomingIds = incoming.map { it.first }.toSet()
        val photoLookup = photoRepository.findAllById(incomingIds).associateBy { it.id }
        val eventLookup = eventRepository.findAllById(incoming.map { it.second }.toSet())
            .associateBy { it.id }
        for ((photoId, eventId) in incoming) {
            val photo = photoLookup[photoId] ?: continue
            if (photo.eventId != eventId) continue
            if (photo.status != PhotoStatus.LIVE) continue
            // A free photo (V46) would 409 every checkout after login — skip.
            if (photo.pricePhp.signum() <= 0) continue
            val event = eventLookup[eventId] ?: continue
            if (event.status == EventStatus.ARCHIVED) continue
            val key = CartItemId(userId, photoId)
            val row = existing[photoId]
            if (row == null) {
                val saved = cartItemRepository.save(
                    CartItemEntity(
                        id = key,
                        eventId = eventId,
                        pricePhpAtAdd = photo.pricePhp,
                    ),
                )
                existing[photoId] = saved
            } else {
                row.pricePhpAtAdd = photo.pricePhp
                row.eventId = eventId
                cartItemRepository.save(row)
            }
        }
        return list(userId)
    }

    private fun hydrate(rows: List<CartItemEntity>): List<CartItemDto> {
        if (rows.isEmpty()) return emptyList()
        val photoIds = rows.map { it.id.photoId }.toSet()
        val eventIds = rows.map { it.eventId }.toSet()
        val photos = photoRepository.findAllById(photoIds).associateBy { it.id }
        val events = eventRepository.findAllById(eventIds).associateBy { it.id }
        return rows.mapNotNull { row ->
            val photo = photos[row.id.photoId] ?: return@mapNotNull null
            val event = events[row.eventId]
            row.toDto(photo, event)
        }
    }

    // `price` is the LIVE `photos.price_php`, not the `pricePhpAtAdd` snapshot.
    // Checkout has no snapshot to honour — `OrderService.create` totals and
    // charges `photos.price_php` — so rendering the snapshot here meant an
    // admin re-price (AdminEventService → PhotoRepository.updatePriceByEventId)
    // left the runner looking at ₱125 and paying ₱150.
    private fun CartItemEntity.toDto(photo: Photo, event: Event?): CartItemDto = CartItemDto(
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
    )

    private fun thumbnailUrlOf(photo: Photo): String {
        val key = photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    private companion object {
        val TIME_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm")
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
    }
}
