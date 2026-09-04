package com.quickpitik.dto.auth

import com.fasterxml.jackson.annotation.JsonInclude
import com.quickpitik.entity.Role
import java.time.OffsetDateTime
import java.util.UUID

// avatarUrl is always wire-presigned; never write a raw S3 URL into the DB.
// Use service.profile.UserDtoMapper to build this — the mapper resolves
// avatarS3Key against StorageService with the avatar TTL.
@JsonInclude(JsonInclude.Include.NON_NULL)
data class UserDto(
    val id: UUID,
    val email: String,
    val name: String,
    val role: Role,
    val avatarUrl: String? = null,
    // Advisory (V30). Additive and non-breaking — an extra key is ignored by
    // the website's TypeScript types and by mobile's Retrofit converter alike.
    // No client reads it yet; nothing on the server gates on it either.
    val emailVerified: Boolean = false,
    val createdAt: OffsetDateTime,
)
