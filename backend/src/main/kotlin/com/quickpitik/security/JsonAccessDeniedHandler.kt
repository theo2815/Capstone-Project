package com.quickpitik.security

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ApiError
import com.quickpitik.common.ApiResponse
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.MediaType
import org.springframework.security.access.AccessDeniedException
import org.springframework.security.web.access.AccessDeniedHandler
import org.springframework.stereotype.Component

@Component
class JsonAccessDeniedHandler(
    private val objectMapper: ObjectMapper,
) : AccessDeniedHandler {
    override fun handle(
        request: HttpServletRequest,
        response: HttpServletResponse,
        accessDeniedException: AccessDeniedException,
    ) {
        response.contentType = MediaType.APPLICATION_JSON_VALUE
        response.status = HttpServletResponse.SC_FORBIDDEN
        objectMapper.writeValue(
            response.outputStream,
            ApiResponse.failure(ApiError(code = "FORBIDDEN", message = "Access denied")),
        )
    }
}
