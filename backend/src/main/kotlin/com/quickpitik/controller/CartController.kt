package com.quickpitik.controller

import com.quickpitik.dto.cart.AddCartItemRequest
import com.quickpitik.dto.cart.CartItemDto
import com.quickpitik.dto.cart.ClearedResponse
import com.quickpitik.dto.cart.MergeCartRequest
import com.quickpitik.dto.cart.RemovedResponse
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.cart.CartService
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/cart")
class CartController(
    private val cartService: CartService,
) {
    @GetMapping
    fun list(@AuthenticationPrincipal principal: AuthPrincipal): List<CartItemDto> =
        cartService.list(principal.userId)

    @PostMapping("/items")
    fun add(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestBody body: AddCartItemRequest,
    ): CartItemDto =
        cartService.add(principal.userId, body.photoId, body.eventId)

    @DeleteMapping("/items/{photoId}")
    fun remove(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable photoId: UUID,
    ): RemovedResponse =
        RemovedResponse(removed = cartService.remove(principal.userId, photoId))

    @PostMapping("/merge")
    fun merge(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestBody body: MergeCartRequest,
    ): List<CartItemDto> =
        cartService.merge(
            userId = principal.userId,
            incoming = body.items.map { it.photoId to it.eventId },
        )

    @DeleteMapping
    fun clear(@AuthenticationPrincipal principal: AuthPrincipal): ClearedResponse =
        ClearedResponse(cleared = cartService.clear(principal.userId))
}
