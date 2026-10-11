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
        if (!token.isActive(now) && !withinRotationGrace(token, now)) {
            throw UnauthorizedException("Refresh token expired or revoked", "INVALID_REFRESH_TOKEN")
        }
        val (child, childPlaintext) = newToken(token.userId)
        // Inside the grace window the parent is left as it is: revokedAt and
        // replacedById stay on the first rotation, so the window runs from
        // there and the grace stays tied to the first child. That child is
        // not revoked: a second browser tab may hold it (two tabs refreshing
        // at once look the same as a lost answer).
        // ponytail: any number of reuses inside the window, each a live
        // token; add a reuse counter or a token family if that must be capped.
        if (token.revokedAt == null) {
            token.revokedAt = now
            token.replacedById = child.id
            refreshTokenRepository.save(token)
        }
        refreshTokenRepository.save(child)
        return token.userId to childPlaintext
    }

    // The child must still be live: a password change or "revoke all" kills
    // it, and the parent's grace with it.
    private fun withinRotationGrace(token: RefreshToken, now: OffsetDateTime): Boolean {
        val revokedAt = token.revokedAt ?: return false
        val childId = token.replacedById ?: return false
        return token.expiresAt.isAfter(now) &&
            revokedAt.plus(jwtProperties.refreshRotationGrace).isAfter(now) &&
            refreshTokenRepository.findById(childId).map { it.revokedAt == null }.orElse(false)
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
        // The kept token stays live, so a parent rotated into it moments ago
        // would keep its grace through the password change: cut the link.
        // ponytail: unindexed lookup, once per password change; index
        // replaced_by_id if refresh_tokens grows large.
        refreshTokenRepository.findByTokenHash(keepHash)
            ?.let { refreshTokenRepository.findByReplacedById(it.id) }
            ?.let { it.replacedById = null; refreshTokenRepository.save(it) }
    }
}
