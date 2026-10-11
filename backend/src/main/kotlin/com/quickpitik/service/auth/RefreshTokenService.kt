package com.quickpitik.service.auth

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
        val (token, plaintext) = newToken(userId)
        refreshTokenRepository.save(token)
        return plaintext
    }

    // A fresh row and its plaintext, not yet saved: rotation links the parent
    // to the row's id first, then saves parent and child in that order (the
    // existing test reads the child as the last save).
    private fun newToken(userId: UUID): Pair<RefreshToken, String> {
        val plaintext = OpaqueTokens.generate()
        val entity = RefreshToken(
            userId = userId,
            tokenHash = OpaqueTokens.hash(plaintext),
            expiresAt = OffsetDateTime.now().plus(jwtProperties.refreshTokenTtl),
        )
        return entity to plaintext
    }

    fun validateAndRotate(plaintext: String): Pair<UUID, String> {
        val hash = OpaqueTokens.hash(plaintext)
        val token = refreshTokenRepository.findByTokenHash(hash)
            ?: throw UnauthorizedException("Invalid refresh token", "INVALID_REFRESH_TOKEN")
        val now = OffsetDateTime.now()
        if (!token.isActive(now)) {
            if (!withinRotationGrace(token, now)) {
                throw UnauthorizedException("Refresh token expired or revoked", "INVALID_REFRESH_TOKEN")
            }
            // The rotation's answer was lost: the child nobody received dies
            // and the parent is rotated once more. revokedAt is kept, so the
            // window runs from the first rotation, not from each reuse.
            // ponytail: any number of reuses inside the window; add a
            // reuse counter if one grace per rotation is ever required.
            token.replacedById?.let { childId ->
                refreshTokenRepository.findById(childId).orElse(null)
                    ?.takeIf { it.revokedAt == null }
                    ?.let { child -> child.revokedAt = now; refreshTokenRepository.save(child) }
            }
        } else {
            token.revokedAt = now
        }
        val (child, childPlaintext) = newToken(token.userId)
        token.replacedById = child.id
        refreshTokenRepository.save(token)
        refreshTokenRepository.save(child)
        return token.userId to childPlaintext
    }

    private fun withinRotationGrace(token: RefreshToken, now: OffsetDateTime): Boolean {
        val revokedAt = token.revokedAt ?: return false
        return token.replacedById != null &&
            token.expiresAt.isAfter(now) &&
            revokedAt.plus(jwtProperties.refreshRotationGrace).isAfter(now)
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
