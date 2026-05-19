package com.quickpitik.controller

import com.quickpitik.dto.orders.OrderDetailDto
import com.quickpitik.dto.orders.OrderStatusDto
import com.quickpitik.service.orders.OrderBundleService
import com.quickpitik.service.orders.OrderService
import org.springframework.http.HttpHeaders
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
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
    private val orderBundleService: OrderBundleService,
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

    // Token-only because <a download> can't carry the JWT — same endpoint
    // serves runners and guests. Response is a streamed body via
    // StreamingResponseBody, which Spring routes through a return-value
    // handler that bypasses ResponseEnvelopeAdvice (otherwise the advice
    // would try to wrap the binary stream in a JSON envelope and corrupt it).
    //
    // Content type + filename come from BundleSpec, which auto-collapses
    // single-photo orders to a raw image/jpeg download instead of a ZIP —
    // the URL is the same; the body and Content-Disposition adapt.
    @GetMapping("/{id}/download-bundle")
    fun bundle(
        @PathVariable id: UUID,
        @RequestParam(required = false) token: String?,
    ): ResponseEntity<StreamingResponseBody> {
        val spec = orderBundleService.prepare(orderId = id, token = token)
        val encoded = URLEncoder.encode(spec.filename, StandardCharsets.UTF_8).replace("+", "%20")
        val body = StreamingResponseBody { out -> orderBundleService.writeTo(spec, out) }
        return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_TYPE, spec.contentType)
            .header(
                HttpHeaders.CONTENT_DISPOSITION,
                """attachment; filename="${spec.filename}"; filename*=UTF-8''$encoded""",
            )
            .header(HttpHeaders.CACHE_CONTROL, "no-store")
            .body(body)
    }
}
