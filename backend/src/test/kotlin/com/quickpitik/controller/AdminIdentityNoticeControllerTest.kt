package com.quickpitik.controller

import com.quickpitik.dto.admin.IdentityNoticeResultDto
import com.quickpitik.entity.Role
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.IdentityNoticeService
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.aop.framework.ProxyFactory
import org.springframework.security.access.AccessDeniedException
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.authorization.method.AuthorizationManagerBeforeMethodInterceptor
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import java.util.UUID
import kotlin.test.assertFailsWith

// The route mails real photographers, so a call without dryRun must only count.
// Same secured proxy as AdminDecisionsControllerTest: class-level @PreAuthorize runs.
class AdminIdentityNoticeControllerTest {

    private val service = Mockito.mock(IdentityNoticeService::class.java)
    private val adminId = UUID.randomUUID()

    private val secured = ProxyFactory(AdminIdentityNoticeController(service)).apply {
        isProxyTargetClass = true
        addAdvice(AuthorizationManagerBeforeMethodInterceptor.preAuthorize())
    }.proxy as AdminIdentityNoticeController

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

    @Test
    fun `a call without dryRun only counts`() {
        Mockito.`when`(service.send(adminId, true)).thenReturn(IdentityNoticeResultDto(3, 0))
        signIn(Role.ADMIN)

        mvc.perform(post("/api/v1/admin/identity-notice")).andExpect(status().isOk)

        Mockito.verify(service).send(adminId, true)
        Mockito.verify(service, Mockito.never()).send(adminId, false)
    }

    @Test
    fun `dryRun=false is passed through`() {
        Mockito.`when`(service.send(adminId, false)).thenReturn(IdentityNoticeResultDto(3, 3))
        signIn(Role.ADMIN)

        mvc.perform(post("/api/v1/admin/identity-notice").param("dryRun", "false")).andExpect(status().isOk)

        Mockito.verify(service).send(adminId, false)
    }

    @Test
    fun `a non-admin is refused before the service runs`() {
        for (role in listOf(Role.RUNNER, Role.PHOTOGRAPHER)) {
            signIn(role)
            assertFailsWith<AccessDeniedException> { secured.send(AuthPrincipal(adminId, "u@test.local", role), true) }
        }
        Mockito.verifyNoInteractions(service)
    }
}
