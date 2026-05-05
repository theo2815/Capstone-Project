package com.quickpitik.service

import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service

@Service
class EmailService(
    @Value("\${app.cors.allowed-origins:http://localhost:3000}") private val frontendOriginsCsv: String,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val frontendOrigin: String =
        frontendOriginsCsv.split(",").firstOrNull()?.trim().orEmpty().ifBlank { "http://localhost:3000" }

    fun sendPasswordResetEmail(toEmail: String, resetToken: String) {
        val resetUrl = "$frontendOrigin/reset-password?token=$resetToken"
        log.info("[EMAIL STUB] password reset for {} — link: {}", toEmail, resetUrl)
    }
}
