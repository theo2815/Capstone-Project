package com.quickpitik.dto.orders

import com.quickpitik.entity.PhotographerCoupon
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// GET/PUT /me/photographer/coupon — mirrors website PhotographerCoupon.
data class CouponDto(
    val code: String,
    val percentOff: Int,
    val active: Boolean,
    val expiresAt: OffsetDateTime?,
    val updatedAt: OffsetDateTime,
)

data class UpsertCouponRequest(
    @field:NotBlank(message = "code is required")
    val code: String,
    val percentOff: Int,
    val active: Boolean = true,
    val expiresAt: OffsetDateTime? = null,
)

// POST /coupons/preview — the checkout modal asks which of the cart's photos a
// code covers and by how much. The same service method that prices a real
// checkout answers here, so the preview can never disagree with the charge.
data class CouponPreviewRequest(
    @field:NotBlank(message = "code is required")
    val code: String,
    @field:NotEmpty(message = "photoIds must not be empty")
    val photoIds: List<UUID> = emptyList(),
)

data class CouponPreviewItemDto(
    val photoId: UUID,
    val price: BigDecimal,
    val discount: BigDecimal,
)

// `items` holds eligible photos only; anything absent is not covered.
data class CouponPreviewDto(
    val code: String,
    val percentOff: Int,
    val photographerName: String?,
    val photographerHandle: String?,
    val items: List<CouponPreviewItemDto>,
    val eligibleCount: Int,
    val discountTotal: BigDecimal,
)

fun PhotographerCoupon.toDto(): CouponDto = CouponDto(
    code = code,
    percentOff = percentOff,
    active = active,
    expiresAt = expiresAt,
    updatedAt = updatedAt,
)
