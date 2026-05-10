package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.CreateSocialRequest
import com.quickpitik.dto.photographer.PatchSocialRequest
import com.quickpitik.dto.photographer.SocialLinkDto
import com.quickpitik.dto.photographer.toDto
import com.quickpitik.entity.SocialLink
import com.quickpitik.entity.SocialPlatform
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.SocialLinkRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.net.URI
import java.util.UUID

@Service
@Transactional
class SocialLinkService(
    private val socialLinkRepository: SocialLinkRepository,
) {
    @Transactional(readOnly = true)
    fun list(userId: UUID): List<SocialLinkDto> =
        socialLinkRepository.findAllByUserIdOrderByCreatedAtAsc(userId).map { it.toDto() }

    fun create(userId: UUID, req: CreateSocialRequest): SocialLinkDto {
        val platform = SocialPlatform.fromWire(req.platform) ?: throw ValidationException(
            code = ErrorCodes.VALIDATION_ERROR,
            message = "platform must be one of facebook|instagram|tiktok|x|youtube|website",
            field = "platform",
        )
        val url = validateUrl(platform, req.url)
        val saved = socialLinkRepository.save(
            SocialLink(userId = userId, platform = platform, url = url),
        )
        return saved.toDto()
    }

    fun patch(userId: UUID, id: UUID, req: PatchSocialRequest): SocialLinkDto {
        val existing = socialLinkRepository.findByIdAndUserId(id, userId)
            ?: throw NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Social link not found")
        existing.url = validateUrl(existing.platform, req.url)
        return socialLinkRepository.save(existing).toDto()
    }

    fun delete(userId: UUID, id: UUID) {
        val existing = socialLinkRepository.findByIdAndUserId(id, userId) ?: return
        socialLinkRepository.delete(existing)
    }

    // Per plan F.2 line 222: server validates host-endsWith per platform;
    // website accepts any https?://. Each platform allows the primary domain
    // plus the well-known shortener domains the FE link normaliser already
    // tolerates (e.g. youtu.be, fb.me).
    private fun validateUrl(platform: SocialPlatform, raw: String): String {
        val trimmed = raw.trim()
        if (!HTTP_PREFIX.containsMatchIn(trimmed)) {
            throw ValidationException(
                code = ErrorCodes.INVALID_SOCIAL_URL,
                message = "url must start with http:// or https://",
                field = "url",
            )
        }
        val host = runCatching { URI(trimmed).host?.lowercase() }.getOrNull()
            ?: throw ValidationException(
                code = ErrorCodes.INVALID_SOCIAL_URL,
                message = "url is malformed",
                field = "url",
            )
        val allowed = ALLOWED_HOSTS[platform]
        if (allowed != null && allowed.none { host == it || host.endsWith(".$it") }) {
            throw ValidationException(
                code = ErrorCodes.INVALID_SOCIAL_URL,
                message = "url host must match ${platform.wire}",
                field = "url",
            )
        }
        return trimmed
    }

    private companion object {
        val HTTP_PREFIX = Regex("^https?://", RegexOption.IGNORE_CASE)
        val ALLOWED_HOSTS: Map<SocialPlatform, Set<String>> = mapOf(
            SocialPlatform.FACEBOOK to setOf("facebook.com", "fb.com", "fb.me"),
            SocialPlatform.INSTAGRAM to setOf("instagram.com", "instagr.am"),
            SocialPlatform.TIKTOK to setOf("tiktok.com"),
            SocialPlatform.X to setOf("x.com", "twitter.com"),
            SocialPlatform.YOUTUBE to setOf("youtube.com", "youtu.be"),
            // WEBSITE deliberately omitted — any https?:// host is accepted.
        )
    }
}
