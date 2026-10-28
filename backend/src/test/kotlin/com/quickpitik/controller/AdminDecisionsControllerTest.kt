package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.dto.admin.AdminDecisionDto
import com.quickpitik.exception.GlobalExceptionHandler
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
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status
import org.springframework.test.web.servlet.setup.MockMvcBuilders
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

    // Same secured proxy behind the real HTTP stack, so the 403 mapping and the body are covered too.
    private val mvc: MockMvc = MockMvcBuilders.standaloneSetup(secured)
        .setControllerAdvice(GlobalExceptionHandler())
        .build()

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
        val page = PaginatedResponse(listOf(row), 1L, 0, 5)
        Mockito.`when`(service.recent(5)).thenReturn(page)
        signIn("ADMIN")

        assertEquals(page, secured.recent(5))
    }

    @Test
    fun `over HTTP a non-admin gets 403 with the access denied envelope`() {
        for (role in listOf("RUNNER", "PHOTOGRAPHER")) {
            signIn(role)
            mvc.perform(get("/api/v1/admin/decisions"))
                .andExpect(status().isForbidden)
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.errors[0].code").value("FORBIDDEN"))
        }
        Mockito.verifyNoInteractions(service)
    }

    @Test
    fun `over HTTP an admin gets 200 with the paged items`() {
        val row = AdminDecisionDto(UUID.randomUUID(), "approved", null, OffsetDateTime.now(), "user", "u1", "Pat", UUID.randomUUID(), "Boss")
        Mockito.`when`(service.recent(5)).thenReturn(PaginatedResponse(listOf(row), 1L, 0, 5))
        signIn("ADMIN")

        mvc.perform(get("/api/v1/admin/decisions").param("limit", "5"))
            .andExpect(status().isOk)
            .andExpect(jsonPath("$.items[0].id").value(row.id.toString()))
            .andExpect(jsonPath("$.items[0].decision").value("approved"))
            .andExpect(jsonPath("$.total").value(1))
            .andExpect(jsonPath("$.offset").value(0))
            .andExpect(jsonPath("$.limit").value(5))
        Mockito.verify(service).recent(5)
    }
}
