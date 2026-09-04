package com.quickpitik.controller

import com.quickpitik.dto.orders.CouponPreviewDto
import com.quickpitik.dto.orders.CouponPreviewRequest
import com.quickpitik.service.orders.CouponService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.ratelimit.clientIp
import jakarta.servlet.http.HttpServletRequest
import jakarta.validation.Valid
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/coupons")
class CouponController(
    private val couponService: CouponService,
    private val rateLimiter: RateLimiter,
) {
    // Guest-allowed (SecurityConfig permits POST /coupons/preview): the code is
    // printed on the photo cards, so anyone at checkout may ask which of their
    // photos it covers. Priced by the same service method OrderService uses,
    // so the modal can never promise a total the charge won't honour.
    @PostMapping("/preview")
    fun preview(
        @Valid @RequestBody body: CouponPreviewRequest,
        request: HttpServletRequest,
    ): CouponPreviewDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_COUPON_PREVIEW, clientIp(request))
        return couponService.preview(body)
    }
}
