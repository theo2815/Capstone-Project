package com.quickpitik.dto.auth

import com.fasterxml.jackson.annotation.JsonInclude
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import java.time.OffsetDateTime
import java.util.UUID

@JsonInclude(JsonInclude.Include.NON_NULL)
data class UserDto(
    val id: UUID,
    val email: String,
    val name: String,
    val role: Role,
    val avatarUrl: String? = null,
    val createdAt: OffsetDateTime,
)

fun User.toDto(): UserDto = UserDto(
    id = id,
    email = email,
    name = name,
    role = role,
    avatarUrl = avatarUrl,
    createdAt = createdAt,
)
