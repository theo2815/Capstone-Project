package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.CreateSocialRequest
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.SocialLinkRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.net.URI
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Settles the 2026-05-27 audit's "social URL path-injection" finding, which
// claimed `https://instagram.com@attacker.com/` slips past the host allow-list
// because SocialLinkService validates the host but not the path.
//
// It does not. RFC-3986 userinfo means java.net.URI parses `instagram.com` as
// USERINFO and `attacker.com` as the host — so the allow-list rejects it on the
// host check alone. Pinning both the parser assumption and the service
// behaviour here so the claim doesn't get re-filed by a future static review.
class SocialLinkUrlTest {

    private val repository: SocialLinkRepository = Mockito.mock(SocialLinkRepository::class.java)
    private fun service() = SocialLinkService(repository)

    @Test
    fun `java-net-URI resolves the host past a userinfo prefix`() {
        val uri = URI("https://instagram.com@attacker.com/")
        assertEquals("attacker.com", uri.host)
        assertEquals("instagram.com", uri.userInfo)
    }

    @Test
    fun `the userinfo-prefixed lookalike URL is rejected`() {
        val ex = assertFailsWith<ValidationException> {
            service().create(
                userId = java.util.UUID.randomUUID(),
                req = CreateSocialRequest(
                    platform = "instagram",
                    url = "https://instagram.com@attacker.com/",
                ),
            )
        }

        assertEquals(ErrorCodes.INVALID_SOCIAL_URL, ex.code)
        Mockito.verifyNoInteractions(repository)
    }

    @Test
    fun `a lookalike suffix domain is rejected`() {
        val ex = assertFailsWith<ValidationException> {
            service().create(
                userId = java.util.UUID.randomUUID(),
                req = CreateSocialRequest(platform = "instagram", url = "https://evil-instagram.com/me"),
            )
        }

        assertEquals(ErrorCodes.INVALID_SOCIAL_URL, ex.code)
    }
}
