package com.quickpitik.mobile.ui.common

import android.net.Uri
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * The deletion request opens the user's mail app with the address, subject and
 * their account email filled in. An email like "a+b&c@x.com" must survive the
 * trip: unencoded, "&" ends the body early and "+" turns into a space.
 * The link is read the RFC 6068 way mail apps use: split on "?", "&" and "=",
 * then decode. Robolectric for android.net.Uri.
 */
@RunWith(RobolectricTestRunner::class)
class LegalTest {

    @Test
    fun `deletion request fills address, subject and the account email`() {
        val mail = parseMailto(deletionRequestUri("ana+runs&co@example.com"))

        assertEquals(SUPPORT_EMAIL, mail.getValue("to"))
        assertEquals("Account deletion request", mail.getValue("subject"))
        assertTrue(mail.getValue("body"), mail.getValue("body").contains("ana+runs&co@example.com"))
    }

    @Test
    fun `deletion request still opens without a known account email`() {
        val mail = parseMailto(deletionRequestUri(null))

        assertEquals(SUPPORT_EMAIL, mail.getValue("to"))
        assertEquals("Account deletion request", mail.getValue("subject"))
    }

    @Test
    fun `legal pages live on the website`() {
        assertEquals("$WEBSITE_ORIGIN/privacy", PRIVACY_URL)
        assertEquals("$WEBSITE_ORIGIN/terms", TERMS_URL)
        assertTrue(WEBSITE_ORIGIN.startsWith("https://"))
    }

    private fun parseMailto(link: String): Map<String, String> {
        val (address, query) = link.removePrefix("mailto:").split("?", limit = 2)
        return query.split("&").associate {
            val (name, value) = it.split("=", limit = 2)
            name to Uri.decode(value)
        } + ("to" to Uri.decode(address))
    }

    // Same words as the website: the notice also sits on the profile library,
    // which serves every event (privacy browser pass finding 2, 2026-11-07).
    @Test
    fun `face notice names no single event`() {
        assertTrue(FACE_NOTICE.contains("with faces in the photos of the event you search."))
        assertTrue(!FACE_NOTICE.contains("this event's photos"))
    }
}
