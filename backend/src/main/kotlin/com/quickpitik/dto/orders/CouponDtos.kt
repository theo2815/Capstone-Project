package com.quickpitik.dto.orders

import com.quickpitik.entity.PhotographerCoupon
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// GET/PUT /me/photographer/events/{eventId}/coupon.
data class CouponDto(
    val eventId: UUID,
    val code: String,
    val percentOff: Int,
    val active: Boolean,
    val expiresAt: OffsetDateTime?,
    val usageLimit: Int?,
    val usageCount: Long,
    val updatedAt: OffsetDateTime,
)

data class UpsertCouponRequest(
    @field:NotBlank(message = "code is required")
    val code: String,
    val percentOff: Int,
    val active: Boolean = true,
    val expiresAt: OffsetDateTime? = null,
    val usageLimit: Int? = null,
)

// POST /coupons/preview — the checkout modal asks how the cart will be priced.
// Every photographer's live coupon applies to their own photos automatically;
// `code` is an optional typed override. The same predicate that prices a real
// checkout answers here, so the preview can never disagree with the charge.
data class CouponPreviewRequest(
    val code: String? = null,
    @field:NotEmpty(message = "photoIds must not be empty")
    @field:Size(max = 100, message = "photoIds must contain at most 100 photos")
    val photoIds: List<UUID> = emptyList(),
)

data class CouponPreviewItemDto(
    val photoId: UUID,
    val price: BigDecimal,
    val discount: BigDecimal,
    val couponCode: String,
    val percentOff: Int,
)

// `items` holds discounted photos only; anything absent is full price. The
// top-level code fields describe the typed code, null when none was typed.
data class CouponPreviewDto(
    val code: String?,
    val percentOff: Int?,
    val photographerName: String?,
    val photographerHandle: String?,
    val items: List<CouponPreviewItemDto>,
    val eligibleCount: Int,
    val discountTotal: BigDecimal,
)

fun PhotographerCoupon.toDto(usageCount: Long): CouponDto = CouponDto(
    eventId = requireNotNull(eventId),
    code = code,
    percentOff = percentOff,
    active = active,
    expiresAt = expiresAt,
    usageLimit = usageLimit,
    usageCount = usageCount,
    updatedAt = updatedAt,
)
