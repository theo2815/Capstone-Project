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
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerCouponRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
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
) {
    @Transactional(readOnly = true)
    fun get(photographerId: UUID): CouponDto? =
        couponRepository.findById(photographerId).orElse(null)?.toDto()

    fun upsert(photographerId: UUID, req: UpsertCouponRequest): CouponDto {
        val code = normalise(req.code)
        if (!CODE_PATTERN.matches(code)) {
            throw ValidationException(
                message = "code must be 4–16 letters or digits",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "code",
            )
        }
        val max = platformProperties.couponMaxPercent
        if (req.percentOff !in 1..max) {
            throw ValidationException(
                message = "percentOff must be between 1 and $max",
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
        if (couponRepository.existsByCodeAndPhotographerIdNot(code, photographerId)) {
            throw ConflictException(
                message = "That code is already taken by another photographer",
                code = ErrorCodes.COUPON_CODE_TAKEN,
            )
        }
        val coupon = couponRepository.findById(photographerId).orElse(null)?.apply {
            this.code = code
            percentOff = req.percentOff
            active = req.active
            expiresAt = req.expiresAt
            updatedAt = now
        } ?: PhotographerCoupon(
            photographerId = photographerId,
            code = code,
            percentOff = req.percentOff,
            active = req.active,
            expiresAt = req.expiresAt,
        )
        return couponRepository.save(coupon).toDto()
    }

    // Silent-idempotent, like SocialLinkService.delete.
    fun delete(photographerId: UUID) {
        val existing = couponRepository.findById(photographerId).orElse(null) ?: return
        couponRepository.delete(existing)
    }

    // Live coupons for a page of photos, one IN query — the photo DTO resolvers
    // batch this the same way they batch photographer attribution.
    @Transactional(readOnly = true)
    fun activeFor(photographerIds: Set<UUID>): Map<UUID, PhotographerCoupon> {
        if (photographerIds.isEmpty()) return emptyMap()
        val now = OffsetDateTime.now()
        return couponRepository.findAllById(photographerIds)
            .filter { it.isLive(now) }
            .associateBy { it.photographerId }
    }

    @Transactional(readOnly = true)
    fun resolveForCheckout(raw: String): PhotographerCoupon {
        val coupon = couponRepository.findByCode(normalise(raw))?.takeIf { it.active }
            ?: throw ValidationException(
                message = "That coupon code isn't valid",
                code = ErrorCodes.COUPON_INVALID,
                field = "couponCode",
            )
        if (coupon.expiresAt?.isBefore(OffsetDateTime.now()) == true) {
            throw ValidationException(
                message = "That coupon code has expired",
                code = ErrorCodes.COUPON_EXPIRED,
                field = "couponCode",
            )
        }
        return coupon
    }

    // Free (₱0) photos are never eligible — there is no share to discount.
    fun eligible(photo: Photo, coupon: PhotographerCoupon): Boolean =
        photo.photographerId == coupon.photographerId && photo.pricePhp.signum() > 0

    fun discountFor(photo: Photo, coupon: PhotographerCoupon): BigDecimal =
        couponDiscount(photo.pricePhp, platformProperties.photographerKeepRate, coupon.percentOff)

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

    @Transactional(readOnly = true)
    fun preview(req: CouponPreviewRequest): CouponPreviewDto {
        val coupon = resolveForCheckout(req.code)
        val items = photoRepository.findAllById(req.photoIds.distinct())
            .filter { it.status == PhotoStatus.LIVE && eligible(it, coupon) }
            .map { CouponPreviewItemDto(photoId = it.id, price = it.pricePhp, discount = discountFor(it, coupon)) }
        val name = userRepository.findById(coupon.photographerId).orElse(null)?.name
        if (items.isEmpty()) {
            throw ValidationException(
                message = "${coupon.code} belongs to ${name ?: "another photographer"}; none of these photos are theirs",
                code = ErrorCodes.COUPON_NOT_APPLICABLE,
                field = "couponCode",
            )
        }
        val handle = photographerSettingsRepository.findById(coupon.photographerId).orElse(null)?.handle
        return CouponPreviewDto(
            code = coupon.code,
            percentOff = coupon.percentOff,
            photographerName = name,
            photographerHandle = handle,
            items = items,
            eligibleCount = items.size,
            discountTotal = items.fold(BigDecimal.ZERO) { sum, item -> sum.add(item.discount) },
        )
    }

    companion object {
        fun normalise(raw: String): String = raw.trim().uppercase()
        private val CODE_PATTERN = Regex("^[A-Z0-9]{4,16}$")
    }
}
