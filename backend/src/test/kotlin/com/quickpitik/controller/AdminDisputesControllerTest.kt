package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.admin.AdminDisputeDto
import com.quickpitik.dto.admin.DisputeOrderSnapshotDto
import com.quickpitik.dto.admin.DisputePhotoSnapshotDto
import com.quickpitik.entity.Role
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.exception.ValidationException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminDisputeService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.aop.framework.ProxyFactory
import org.springframework.http.MediaType
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.authorization.method.AuthorizationManagerBeforeMethodInterceptor
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.ResultActions
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import java.math.BigDecimal
import java.time.Duration
import java.util.UUID

// Same secured-proxy-behind-MockMvc shape as AdminDecisionsControllerTest: the real
// class-level @PreAuthorize runs, and the envelope/status mapping is the real handler.
class AdminDisputesControllerTest {

    private val service = Mockito.mock(AdminDisputeService::class.java)
    private val limiter = Mockito.mock(RateLimiter::class.java)
    private val adminId = UUID.randomUUID()
    private val disputeId = UUID.randomUUID()
    private val url = "https://transfer.test/abc"

    private val secured = ProxyFactory(AdminDisputesController(service, limiter)).apply {
        isProxyTargetClass = true
        addAdvice(AuthorizationManagerBeforeMethodInterceptor.preAuthorize())
    }.proxy as AdminDisputesController

    private val mvc: MockMvc = MockMvcBuilders.standaloneSetup(secured)
        .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver())
        .setControllerAdvice(GlobalExceptionHandler())
        .build()

    @AfterEach
    fun clear() = SecurityContextHolder.clearContext()

    private fun signIn(role: Role) {
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(
            AuthPrincipal(adminId, "u@test.local", role), "n/a", listOf(SimpleGrantedAuthority("ROLE_${role.name}")),
        )
    }

    private fun allowRate() {
        Mockito.`when`(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, adminId.toString()))
            .thenReturn(RateLimiter.Decision(true))
    }

    private fun sendLink(body: String = "{\"url\":\"$url\"}"): ResultActions =
        mvc.perform(
            post("/api/v1/admin/disputes/{id}/refund-link", disputeId)
                .contentType(MediaType.APPLICATION_JSON).content(body),
        )

    private fun dto() = AdminDisputeDto(
        id = disputeId, orderId = UUID.randomUUID(), photoId = UUID.randomUUID(), eventId = UUID.randomUUID(),
        eventName = null, runnerHandle = "r", photographerHandle = "p", reason = "other", note = "", status = "escalated",
        reportedAt = "2026-10-29T00:00:00Z", resolvedAt = null, refundAmount = BigDecimal("127.50"),
        refundStatus = "manual_review", paymentId = null, paymentRef = null, refundLinkUrl = url,
        refundLinkSentAt = "2026-10-29T00:00:00Z", itemChargeAmount = null, resolution = "refund_full",
        orderSnapshot = DisputeOrderSnapshotDto(BigDecimal("127.50"), "qrph", null),
        photoSnapshot = DisputePhotoSnapshotDto("", null, null, null),
    )

    @Test
    fun `a non-admin cannot send a refund link`() {
        for (role in listOf(Role.RUNNER, Role.PHOTOGRAPHER)) {
            signIn(role)
            sendLink().andExpect(status().isForbidden)
        }
        Mockito.verifyNoInteractions(service)
        Mockito.verifyNoInteractions(limiter)
    }

    @Test
    fun `an admin sends the link through the service after taking a rate-limit slot`() {
        signIn(Role.ADMIN)
        allowRate()
        Mockito.`when`(service.sendRefundLink(adminId, disputeId, url)).thenReturn(dto())

        sendLink().andExpect(status().isOk).andExpect(jsonPath("$.id").value(disputeId.toString()))

        Mockito.verify(limiter).tryAcquire(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, adminId.toString())
        Mockito.verify(service).sendRefundLink(adminId, disputeId, url)
    }

    @Test
    fun `a link the service refuses as a bad prefix is a 400`() {
        signIn(Role.ADMIN)
        allowRate()
        Mockito.`when`(service.sendRefundLink(adminId, disputeId, url))
            .thenThrow(ValidationException("Not an allowed refund link", field = "url"))

        sendLink().andExpect(status().isBadRequest).andExpect(jsonPath("$.errors[0].code").value(ErrorCodes.VALIDATION_ERROR))
    }

    @Test
    fun `a dispute that is not waiting for a manual refund is a 409`() {
        signIn(Role.ADMIN)
        allowRate()
        Mockito.`when`(service.sendRefundLink(adminId, disputeId, url))
            .thenThrow(ConflictException(message = "Not in manual review", code = ErrorCodes.CONFLICT))

        sendLink().andExpect(status().isConflict)
    }

    @Test
    fun `a blank link is a 400 before the service runs`() {
        signIn(Role.ADMIN)
        allowRate()

        sendLink("{\"url\":\"  \"}").andExpect(status().isBadRequest)

        Mockito.verifyNoInteractions(service)
    }

    private fun markRefunded(): ResultActions =
        mvc.perform(post("/api/v1/admin/disputes/{id}/mark-refunded", disputeId))

    @Test
    fun `a non-admin cannot mark a refund refunded`() {
        for (role in listOf(Role.RUNNER, Role.PHOTOGRAPHER)) {
            signIn(role)
            markRefunded().andExpect(status().isForbidden)
        }
        Mockito.verifyNoInteractions(service)
        Mockito.verifyNoInteractions(limiter)
    }

    @Test
    fun `an admin marks a refund refunded through the service after taking a rate-limit slot`() {
        signIn(Role.ADMIN)
        allowRate()
        Mockito.`when`(service.markRefunded(adminId, disputeId)).thenReturn(dto())

        markRefunded().andExpect(status().isOk).andExpect(jsonPath("$.id").value(disputeId.toString()))

        Mockito.verify(limiter).tryAcquire(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, adminId.toString())
        Mockito.verify(service, Mockito.times(1)).markRefunded(adminId, disputeId)
    }

    @Test
    fun `marking a dispute that is not in manual review refunded is a 409`() {
        signIn(Role.ADMIN)
        allowRate()
        Mockito.`when`(service.markRefunded(adminId, disputeId))
            .thenThrow(ConflictException(message = "Not in manual review", code = ErrorCodes.CONFLICT))

        markRefunded().andExpect(status().isConflict)
    }

    @Test
    fun `a rate-limited admin marking refunded gets 429 and the service is not reached`() {
        signIn(Role.ADMIN)
        Mockito.`when`(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, adminId.toString()))
            .thenReturn(RateLimiter.Decision(false, Duration.ofSeconds(30)))

        markRefunded().andExpect(status().isTooManyRequests)

        Mockito.verifyNoInteractions(service)
    }

    @Test
    fun `a rate-limited admin gets 429 and the service is not reached`() {
        signIn(Role.ADMIN)
        Mockito.`when`(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, adminId.toString()))
            .thenReturn(RateLimiter.Decision(false, Duration.ofSeconds(30)))

        sendLink().andExpect(status().isTooManyRequests)

        Mockito.verifyNoInteractions(service)
    }
}
