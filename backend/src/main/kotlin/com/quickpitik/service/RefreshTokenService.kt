package com.quickpitik.service

import com.quickpitik.config.JwtProperties
import com.quickpitik.entity.RefreshToken
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.repository.RefreshTokenRepository
import com.quickpitik.security.OpaqueTokens
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class RefreshTokenService(
    private val refreshTokenRepository: RefreshTokenRepository,
    private val jwtProperties: JwtProperties,
) {
    fun issue(userId: UUID): String {
        val plaintext = OpaqueTokens.generate()
        val entity = RefreshToken(
            userId = userId,
            tokenHash = OpaqueTokens.hash(plaintext),
            expiresAt = OffsetDateTime.now().plus(jwtProperties.refreshTokenTtl),
        )
        refreshTokenRepository.save(entity)
        return plaintext
    }

    fun validateAndRotate(plaintext: String): Pair<UUID, String> {
        val hash = OpaqueTokens.hash(plaintext)
        val token = refreshTokenRepository.findByTokenHash(hash)
            ?: throw UnauthorizedException("Invalid refresh token", "INVALID_REFRESH_TOKEN")
        if (!token.isActive()) {
            throw UnauthorizedException("Refresh token expired or revoked", "INVALID_REFRESH_TOKEN")
        }
        token.revokedAt = OffsetDateTime.now()
        refreshTokenRepository.save(token)
        val newPlaintext = issue(token.userId)
        return token.userId to newPlaintext
    }

    fun revoke(plaintext: String) {
        val hash = OpaqueTokens.hash(plaintext)
        val token = refreshTokenRepository.findByTokenHash(hash) ?: return
        token.revokedAt = OffsetDateTime.now()
        refreshTokenRepository.save(token)
    }

    fun revokeAllForUser(userId: UUID) {
        refreshTokenRepository.revokeAllForUser(userId, OffsetDateTime.now())
    }

    fun revokeAllForUserExcept(userId: UUID, keepPlaintext: String?) {
        val keepHash = keepPlaintext?.takeIf { it.isNotBlank() }?.let { OpaqueTokens.hash(it) }
        if (keepHash == null) {
            refreshTokenRepository.revokeAllForUser(userId, OffsetDateTime.now())
            return
        }
        refreshTokenRepository.revokeAllForUserExcept(userId, keepHash, OffsetDateTime.now())
    }
}
