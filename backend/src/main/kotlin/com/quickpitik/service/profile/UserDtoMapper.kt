package com.quickpitik.service.profile

import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.entity.User
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service

/**
 * Single point of truth for User → UserDto conversion. Presigns avatar_s3_key
 * with the configured avatar TTL on every read so the FE always receives a
 * fresh URL — no expired URLs ever land in the row.
 *
 * Falls back to the legacy users.avatar_url column when avatar_s3_key is null
 * (pre-PR-6 users have no s3 key — this keeps existing accounts rendering
 * correctly until they upload).
 */
@Service
class UserDtoMapper(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
) {
    fun toDto(user: User): UserDto = UserDto(
        id = user.id,
        email = user.email,
        name = user.name,
        role = user.role,
        avatarUrl = resolveAvatarUrl(user),
        createdAt = user.createdAt,
    )

    fun resolveAvatarUrl(user: User): String? {
        val key = user.avatarS3Key
        if (key.isNullOrBlank()) {
            return user.avatarUrl
        }
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.avatar)
    }
}
