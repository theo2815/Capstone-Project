package com.quickpitik.controller

import com.quickpitik.dto.admin.AdminUserRowDto
import com.quickpitik.entity.Role
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminUserService
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.aop.framework.ProxyFactory
import org.springframework.http.MediaType
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

// R-10: the admin approve route for a held profile. Same secured proxy as AdminIdentityNoticeControllerTest.
class AdminUsersControllerTest {

    private val service = Mockito.mock(AdminUserService::class.java)
    private val adminId = UUID.randomUUID()
    private val userId = UUID.randomUUID()

    private val secured = ProxyFactory(AdminUsersController(service)).apply {
        isProxyTargetClass = true
        addAdvice(AuthorizationManagerBeforeMethodInterceptor.preAuthorize())
    }.proxy as AdminUsersController

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
    fun `approve passes the revision through`() {
        Mockito.`when`(service.approveHeld(adminId, userId, 4)).thenReturn(AdminUserRowDto(
            userId, "PHOTOGRAPHER", "p@test.local", "P", null, null, null, null, "", "", "approved", null, null, null, 5))
        signIn(Role.ADMIN)

        mvc.perform(post("/api/v1/admin/users/$userId/approve")
            .contentType(MediaType.APPLICATION_JSON).content("""{"expectedRevision":4}"""))
            .andExpect(status().isOk)

        Mockito.verify(service).approveHeld(adminId, userId, 4)
    }

    @Test
    fun `a non-admin is refused`() {
        for (role in listOf(Role.RUNNER, Role.PHOTOGRAPHER)) {
            signIn(role)
            assertFailsWith<AccessDeniedException> {
                secured.approve(AuthPrincipal(adminId, "u@test.local", role), userId, null)
            }
        }
        Mockito.verifyNoInteractions(service)
    }
}
