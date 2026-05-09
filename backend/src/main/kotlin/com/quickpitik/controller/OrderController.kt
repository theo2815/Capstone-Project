package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.orders.CreateOrderRequest
import com.quickpitik.dto.orders.OrderDetailDto
import com.quickpitik.dto.orders.OrderListItemDto
import com.quickpitik.dto.orders.OrderResponse
import com.quickpitik.dto.orders.RefundRequest
import com.quickpitik.dto.orders.RefundResponse
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.orders.OrderService
import com.quickpitik.service.orders.RefundService
import jakarta.validation.Valid
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

@RestController
@RequestMapping("/api/v1")
class OrderController(
    private val orderService: OrderService,
    private val refundService: RefundService,
) {
    // Guest-allowed: principal may be null. SecurityConfig permits POST /orders.
    @PostMapping("/orders")
    fun create(
        @AuthenticationPrincipal principal: AuthPrincipal?,
        @Valid @RequestBody body: CreateOrderRequest,
    ): OrderResponse =
        orderService.create(userId = principal?.userId, request = body)

    @GetMapping("/me/orders")
    fun list(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<OrderListItemDto> =
        orderService.listForUser(
            userId = principal.userId,
            params = PaginationParams.of(offset, limit),
        )

    @GetMapping("/me/orders/{id}")
    fun detail(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): OrderDetailDto =
        orderService.getDetail(userId = principal.userId, orderId = id)

    @PostMapping("/me/orders/{id}/refund")
    fun refund(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
        @Valid @RequestBody body: RefundRequest,
    ): RefundResponse =
        refundService.submit(userId = principal.userId, orderId = id, request = body)
}
