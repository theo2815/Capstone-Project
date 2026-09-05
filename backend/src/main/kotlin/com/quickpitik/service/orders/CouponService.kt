package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.orders.CouponDto
import com.quickpitik.dto.orders.CouponPreviewDto
import com.quickpitik.dto.orders.CouponPreviewItemDto
import com.quickpitik.dto.orders.CouponPreviewRequest
import com.quickpitik.dto.orders.UpsertCouponRequest
import com.quickpitik.dto.orders.toDto
import com.quickpitik.dto.photos.CouponQuote
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerCouponRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.OffsetDateTime
import java.util.UUID

// The only place the coupon split is computed. A coupon is a percentage of the
// photographer's share (listPrice × keepRate), never of the list price, so the
// platform fee (listPrice × cutRate) is identical with or without a code.
// TransactionMintingService subtracts this same figure from the kept amount.
fun couponDiscount(listPrice: BigDecimal, keepRate: BigDecimal, percentOff: Int): BigDecimal =
    listPrice.multiply(keepRate)
        .multiply(BigDecimal(percentOff))
        .divide(BigDecimal(100), 2, RoundingMode.HALF_UP)

@Service
@Transactional
class CouponService(
    private val couponRepository: PhotographerCouponRepository,
    private val photoRepository: PhotoRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val platformProperties: PlatformProperties,
    private val eventRepository: EventRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val orderRepository: OrderRepository,
) {
    @Transactional(readOnly = true)
    fun get(photographerId: UUID, eventId: UUID): CouponDto? {
        requireCoveredEvent(photographerId, eventId)
        return couponRepository.findByEventIdAndPhotographerId(eventId, photographerId)
            ?.let { it.toDto(usageCountOf(it)) }
    }

    fun upsert(photographerId: UUID, eventId: UUID, req: UpsertCouponRequest): CouponDto {
        val event = requireEligibleCoverage(photographerId, eventId)
        val code = normalise(req.code)
        if (!CODE_PATTERN.matches(code)) {
            throw ValidationException(
                message = "code must be 4–16 letters or digits",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "code",
            )
        }
        val max = platformProperties.couponMaxPercent
        // Free giveaway (2026-09-05): exactly 100% on a paid event the
        // photographer created zeroes the list price (see discountFor).
        // Covered-but-not-owned and admin events keep the cap.
        val freeGiveaway = req.percentOff == 100 && event.createdBy == photographerId
        if (!freeGiveaway && req.percentOff !in 1..max) {
            throw ValidationException(
                message = "percentOff must be between 1 and $max, or 100 on an event you created",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "percentOff",
            )
        }
        val now = OffsetDateTime.now()
        if (req.expiresAt != null && !req.expiresAt.isAfter(now)) {
            throw ValidationException(
                message = "expiresAt must be in the future",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "expiresAt",
            )
        }
        if (req.usageLimit != null && req.usageLimit !in 1..MAX_USAGE_LIMIT) {
            throw ValidationException(
                message = "usageLimit must be between 1 and $MAX_USAGE_LIMIT",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "usageLimit",
            )
        }
        val coupon = couponRepository.findByEventIdAndPhotographerId(eventId, photographerId)?.apply {
            this.code = code
            percentOff = req.percentOff
            active = req.active
            expiresAt = req.expiresAt
            usageLimit = req.usageLimit
            updatedAt = now
        } ?: PhotographerCoupon(
            eventId = eventId,
            photographerId = photographerId,
            code = code,
            percentOff = req.percentOff,
            active = req.active,
            expiresAt = req.expiresAt,
            usageLimit = req.usageLimit,
        )
        if (couponRepository.existsByCodeAndEventIdIsNotNullAndIdNot(code, coupon.id)) {
            throw ConflictException(
                message = "That code is already taken by another photographer",
                code = ErrorCodes.COUPON_CODE_TAKEN,
            )
        }
        return try {
            couponRepository.saveAndFlush(coupon).toDto(usageCountOf(coupon))
        } catch (_: DataIntegrityViolationException) {
            throw ConflictException(
                message = "That coupon code or event already has a coupon",
                code = ErrorCodes.COUPON_CODE_TAKEN,
            )
        }
    }

    // Silent-idempotent, like SocialLinkService.delete.
    fun delete(photographerId: UUID, eventId: UUID) {
        requireCoveredEvent(photographerId, eventId)
        val existing = couponRepository.findByEventIdAndPhotographerId(eventId, photographerId) ?: return
        couponRepository.delete(existing)
    }

    // Live coupons for a page of photos, one IN query — the photo DTO resolvers
    // batch this the same way they batch photographer attribution.
    @Transactional(readOnly = true)
    fun activeFor(eventId: UUID, photographerIds: Set<UUID>): Map<UUID, PhotographerCoupon> {
        if (photographerIds.isEmpty()) return emptyMap()
        return couponRepository.findLiveForEvent(eventId, photographerIds, OffsetDateTime.now())
            .associateBy { it.photographerId }
    }

    @Transactional(readOnly = true)
    fun resolveForCheckout(raw: String): PhotographerCoupon {
        val coupon = couponRepository.findByCodeAndEventIdIsNotNull(normalise(raw))
        return requireRedeemable(coupon)
    }

    // Checkout calls this inside its reservation transaction. The row lock
    // serializes the usage-count check with creation of the discounted order.
    fun reserveForCheckout(raw: String): PhotographerCoupon {
        val coupon = couponRepository.findScopedByCodeForUpdate(normalise(raw))
        return requireRedeemable(coupon)
    }

    // Auto-apply (2026-09-05): every live coupon of the (event, photographer)
    // pairs in the cart, locked like reserveForCheckout so the usage-limit
    // check serializes with the discounted order's creation. Never throws —
    // a coupon that can't be redeemed simply isn't applied. Keyed by pair, so
    // a photographer's coupon on one event can never reach their photos in
    // another, and one photographer's coupon never reaches another's photos.
    fun reserveAutoFor(photos: Collection<Photo>): Map<Pair<UUID, UUID>, PhotographerCoupon> =
        autoFor(photos, couponRepository::findActiveByEventIdInForUpdate)

    private fun autoFor(
        photos: Collection<Photo>,
        load: (Set<UUID>) -> List<PhotographerCoupon>,
    ): Map<Pair<UUID, UUID>, PhotographerCoupon> {
        val pairs = photos.mapNotNull { p -> p.photographerId?.let { p.eventId to it } }.toSet()
        if (pairs.isEmpty()) return emptyMap()
        val now = OffsetDateTime.now()
        return load(pairs.map { it.first }.toSet())
            .filter { it.eventId != null && it.active && (it.eventId to it.photographerId) in pairs }
            .filter { it.expiresAt?.isAfter(now) != false }
            .filter { c -> c.usageLimit?.let { usageCountOf(c) < it } != false }
            .associateBy { it.eventId!! to it.photographerId }
    }

    // Free (₱0) photos are never eligible — there is no share to discount.
    fun eligible(photo: Photo, coupon: PhotographerCoupon): Boolean =
        photo.eventId == coupon.eventId &&
            photo.photographerId == coupon.photographerId &&
            photo.pricePhp.signum() > 0

    // A 100% giveaway (own paid event only, enforced at upsert) waives the
    // platform cut too: the runner pays ₱0 and nobody earns anything.
    fun discountFor(photo: Photo, coupon: PhotographerCoupon): BigDecimal =
        if (coupon.percentOff == 100) photo.pricePhp
        else couponDiscount(photo.pricePhp, platformProperties.photographerKeepRate, coupon.percentOff)

    // One photo's offer for the DTO layer: null unless the coupon is live for
    // this photo's owner and the photo is priced.
    fun quoteFor(photo: Photo, coupon: PhotographerCoupon?): CouponQuote? {
        if (coupon == null || !eligible(photo, coupon)) return null
        return CouponQuote(
            code = coupon.code,
            percentOff = coupon.percentOff,
            price = photo.pricePhp.subtract(discountFor(photo, coupon)),
        )
    }

    // The typed code wins for its own pair; every other pair keeps its
    // automatic coupon. Same precedence as OrderService.reserveCheckout.
    fun couponFor(photo: Photo, manual: PhotographerCoupon?, auto: Map<Pair<UUID, UUID>, PhotographerCoupon>) =
        manual?.takeIf { eligible(photo, it) }
            ?: photo.photographerId?.let { auto[photo.eventId to it] }?.takeIf { eligible(photo, it) }

    @Transactional(readOnly = true)
    fun preview(req: CouponPreviewRequest): CouponPreviewDto {
        val manual = req.code?.takeIf { it.isNotBlank() }?.let(::resolveForCheckout)
        val photos = photoRepository.findAllById(req.photoIds.distinct()).filter { it.status == PhotoStatus.LIVE }
        // Read-only quote: no lock, and findLiveForEvent already filters
        // active / expiry / usage-limit.
        val auto = autoFor(photos) { eventIds ->
            eventIds.flatMap { eventId ->
                couponRepository.findLiveForEvent(
                    eventId,
                    photos.filter { it.eventId == eventId }.mapNotNull { it.photographerId }.toSet(),
                    OffsetDateTime.now(),
                )
            }
        }
        val items = photos.mapNotNull { photo ->
            couponFor(photo, manual, auto)?.let { coupon ->
                CouponPreviewItemDto(
                    photoId = photo.id,
                    price = photo.pricePhp,
                    discount = discountFor(photo, coupon),
                    couponCode = coupon.code,
                    percentOff = coupon.percentOff,
                )
            }
        }
        val name = manual?.let { userRepository.findById(it.photographerId).orElse(null)?.name }
        if (manual != null && photos.none { eligible(it, manual) }) {
            throw ValidationException(
                message = "${manual.code} belongs to ${name ?: "another photographer"}; none of these photos are theirs",
                code = ErrorCodes.COUPON_NOT_APPLICABLE,
                field = "couponCode",
            )
        }
        val handle = manual?.let { photographerSettingsRepository.findById(it.photographerId).orElse(null)?.handle }
        return CouponPreviewDto(
            code = manual?.code,
            percentOff = manual?.percentOff,
            photographerName = name,
            photographerHandle = handle,
            items = items,
            eligibleCount = items.size,
            discountTotal = items.fold(BigDecimal.ZERO) { sum, item -> sum.add(item.discount) },
        )
    }

    private fun requireRedeemable(coupon: PhotographerCoupon?): PhotographerCoupon {
        val scoped = coupon?.takeIf { it.eventId != null && it.active }
            ?: throw ValidationException(
                message = "That coupon code isn't valid",
                code = ErrorCodes.COUPON_INVALID,
                field = "couponCode",
            )
        if (scoped.expiresAt?.isAfter(OffsetDateTime.now()) == false) {
            throw ValidationException(
                message = "That coupon code has expired",
                code = ErrorCodes.COUPON_EXPIRED,
                field = "couponCode",
            )
        }
        if (scoped.usageLimit?.let { usageCountOf(scoped) >= it.toLong() } == true) {
            throw ValidationException(
                message = "That coupon has reached its usage limit",
                code = ErrorCodes.COUPON_USAGE_LIMIT_REACHED,
                field = "couponCode",
            )
        }
        return scoped
    }

    private fun usageCountOf(coupon: PhotographerCoupon): Long =
        orderRepository.countUsesExcludingStatus(coupon.id, OrderStatus.EXPIRED)

    // "Covered" is the same predicate PhotographerEventService.getEventDetail
    // uses: the photographer created the event (V46) or has an
    // event_photographer row (first upload). Admin events have no creator, so
    // ownership alone would lock out everyone who actually shot them. Uncovered
    // and unknown events answer the same 404 — never leak existence.
    private fun requireCoveredEvent(photographerId: UUID, eventId: UUID): Event {
        val event = eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val covered = event.createdBy == photographerId ||
            eventPhotographerRepository.existsById(EventPhotographerId(eventId, photographerId))
        if (!covered) throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        return event
    }

    private fun requireEligibleCoverage(photographerId: UUID, eventId: UUID): Event {
        val event = requireCoveredEvent(photographerId, eventId)
        val user = userRepository.findById(photographerId).orElse(null)
            ?: throw NotFoundException(code = ErrorCodes.USER_NOT_FOUND, message = "User not found")
        if (user.suspendedAt != null) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.ACCOUNT_SUSPENDED,
                message = "Your account is suspended. Contact support before creating coupons.",
            )
        }
        if (photographerSettingsRepository.findById(photographerId).orElse(null)?.verificationStatus !=
            VerificationStatus.APPROVED
        ) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED,
                message = "Complete photographer verification before creating coupons.",
            )
        }
        if (event.isFree) {
            throw ValidationException(
                message = "Free events cannot use coupons",
                code = ErrorCodes.COUPON_NOT_APPLICABLE,
                field = "eventId",
            )
        }
        return event
    }

    companion object {
        fun normalise(raw: String): String = raw.trim().uppercase()
        private val CODE_PATTERN = Regex("^[A-Z0-9]{4,16}$")
        private const val MAX_USAGE_LIMIT = 1_000_000
    }
}
