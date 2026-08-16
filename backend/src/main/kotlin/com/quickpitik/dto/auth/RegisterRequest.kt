package com.quickpitik.dto.auth

import com.quickpitik.entity.Role
import jakarta.validation.constraints.Email
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size

data class RegisterRequest(
    @field:NotBlank
    @field:Size(min = 1, max = 255)
    val name: String,

    @field:Email
    @field:NotBlank
    val email: String,

    @field:NotBlank
    // 72 = bcrypt's truncation point; see PasswordValidator.MAX_BYTES, which
    // enforces the same ceiling in bytes for multi-byte input.
    @field:Size(min = 8, max = 72, message = "Password must be 8-72 characters")
    val password: String,

    val role: Role,
)
