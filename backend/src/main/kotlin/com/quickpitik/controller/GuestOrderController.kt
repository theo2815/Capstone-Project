package com.quickpitik.controller

import com.quickpitik.dto.orders.OrderDetailDto
import com.quickpitik.dto.orders.OrderStatusDto
import com.quickpitik.service.orders.OrderService
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// Guest-accessible order status. The `/orders/return` page polls this after
// PayMongo redirects the user back, so guests (no JWT) can confirm their
// order flipped to PAID. Authed runners use `/me/orders/{id}` for the full
// hydrated detail; this endpoint stays minimal by design — no item/photo
// URLs leak without a verified share token + email link (Phase 4).
//
// Anti-IDOR: missing/mismatched token returns the same NOT_FOUND envelope as
// a truly nonexistent id. SecurityConfig permits this path; the service
// layer enforces the token check.
@RestController
@RequestMapping("/api/v1/orders")
class GuestOrderController(
    private val orderService: OrderService,
) {
    @GetMapping("/{id}/status")
    fun status(
        @PathVariable id: UUID,
        @RequestParam(required = false) token: String?,
    ): OrderStatusDto = orderService.statusByIdAndToken(orderId = id, token = token)

    // Hydrated detail for the /orders/return success state. Same shape as
    // `/me/orders/{id}` but gated on the share token (anti-IDOR; failures
    // surface as NOT_FOUND, not 401, so guessing-by-id reveals nothing).
    @GetMapping("/{id}")
    fun detail(
        @PathVariable id: UUID,
        @RequestParam(required = false) token: String?,
    ): OrderDetailDto = orderService.detailByIdAndToken(orderId = id, token = token)
}
