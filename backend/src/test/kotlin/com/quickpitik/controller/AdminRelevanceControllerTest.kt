package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.Role
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.RelevanceAppealService
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.MediaType
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import java.util.UUID

// RVF-16 (NIT 1): Postgres refuses a NUL character in TEXT, so an admin's case note carrying one is a 400, not a 500.
// The five case actions share RelevanceActionRequest, so one route stands for all.
class AdminRelevanceControllerTest {
    private val service = Mockito.mock(AdminRelevanceService::class.java)
    private val mvc: MockMvc = MockMvcBuilders.standaloneSetup(AdminRelevanceController(service, Mockito.mock(RelevanceAppealService::class.java)))
        .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver())
        .setControllerAdvice(GlobalExceptionHandler())
        .build()

    @AfterEach
    fun clear() = SecurityContextHolder.clearContext()

    @Test
    fun `an admin note with a NUL character is a 400`() {
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(
            AuthPrincipal(UUID.randomUUID(), "admin@test.local", Role.ADMIN), "n/a", listOf(SimpleGrantedAuthority("ROLE_ADMIN")),
        )

        mvc.perform(post("/api/v1/admin/relevance/cases/{id}/restore", UUID.randomUUID())
            .contentType(MediaType.APPLICATION_JSON).content("{\"note\":\"bad\\u0000note\"}"))
            .andExpect(status().isBadRequest)
            .andExpect(jsonPath("$.errors[0].code").value(ErrorCodes.VALIDATION_ERROR))
            .andExpect(jsonPath("$.errors[0].field").value("note"))
            .andExpect(jsonPath("$.errors[0].message").value("The note cannot contain a NUL character"))
        Mockito.verifyNoInteractions(service)
    }
}
