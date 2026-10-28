package com.quickpitik.controller

import com.quickpitik.dto.admin.AdminDecisionDto
import com.quickpitik.service.admin.AdminDecisionsService
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.aop.framework.ProxyFactory
import org.springframework.security.access.AccessDeniedException
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.authorization.method.AuthorizationManagerBeforeMethodInterceptor
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.context.SecurityContextHolder
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Method security is off in plain unit tests, so this wraps the controller in the
// same pre-authorize advice SecurityConfig turns on: a non-admin must be refused
// before the service is ever reached.
class AdminDecisionsControllerTest {

    private val service = Mockito.mock(AdminDecisionsService::class.java)
    private val secured = ProxyFactory(AdminDecisionsController(service)).apply {
        isProxyTargetClass = true
        addAdvice(AuthorizationManagerBeforeMethodInterceptor.preAuthorize())
    }.proxy as AdminDecisionsController

    @AfterEach
    fun clear() = SecurityContextHolder.clearContext()

    private fun signIn(role: String) {
        SecurityContextHolder.getContext().authentication =
            UsernamePasswordAuthenticationToken("u", "n/a", listOf(SimpleGrantedAuthority("ROLE_$role")))
    }

    @Test
    fun `a non-admin is refused before the service runs`() {
        for (role in listOf("RUNNER", "PHOTOGRAPHER")) {
            signIn(role)
            assertFailsWith<AccessDeniedException> { secured.recent(20) }
        }
        Mockito.verifyNoInteractions(service)
    }

    @Test
    fun `an admin gets the recent decisions with the asked limit`() {
        val row = AdminDecisionDto(UUID.randomUUID(), "approved", null, OffsetDateTime.now(), "user", "u1", "Pat", UUID.randomUUID(), "Boss")
        Mockito.`when`(service.recent(5)).thenReturn(listOf(row))
        signIn("ADMIN")

        assertEquals(listOf(row), secured.recent(5))
    }
}
