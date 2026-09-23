package com.quickpitik.mobile.ui.photographer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The Settings sheet's input rules: which social links it accepts per
 * platform, and how a payout wallet number is grouped and checked. The payout
 * number is where earnings are sent, so a loosened rule is a misrouted payout.
 */
class PhotographerSettingsValidationTest {
    @Test fun `social links must be on the platform's own host`() {
        assertEquals("Add a URL or remove this row.", validateSocialUrl("facebook", "  "))
        assertEquals("URL is at most 2048 characters.", validateSocialUrl("website", "https://a.b/" + "x".repeat(2048)))

        listOf(
            "facebook" to "https://m.facebook.com/juan",
            "facebook" to "https://fb.me/juan",
            "instagram" to "HTTPS://WWW.INSTAGRAM.COM/juan",
            "tiktok" to "https://vm.tiktok.com/abc",
            "x" to "https://twitter.com/juan",
            "x" to "https://x.com/juan",
            "youtube" to "https://youtu.be/abc",
            "youtube" to "https://m.youtube.com/@juan",
            "website" to "  https://juan.photos  ",
            "threads" to "anything", // no rule for an unknown platform
        ).forEach { (platform, url) -> assertNull("$platform $url", validateSocialUrl(platform, url)) }

        assertEquals("This doesn't look like a Instagram URL.", validateSocialUrl("instagram", "https://facebook.com/juan"))
        assertEquals("This doesn't look like a Facebook URL.", validateSocialUrl("facebook", "https://evilfacebook.com/juan"))
        assertEquals("This doesn't look like a Facebook URL.", validateSocialUrl("facebook", "https://facebook.com.evil.io/juan"))
        assertEquals("URL must start with http:// or https://.", validateSocialUrl("website", "juan.photos"))
    }

    @Test fun `wallet numbers are grouped as typed and checked per method`() {
        assertEquals("0917 123 4567", formatPayoutNumber("gcash", "0917-123-4567 99"))
        assertEquals("0917 12", formatPayoutNumber("maya", "091712"))
        assertEquals("0917", formatPayoutNumber("gcash", "0917"))
        assertEquals("1234 5678 9012 3456", formatPayoutNumber("gotyme", "1234567890123456789"))
        assertEquals("1234 56", formatPayoutNumber("gotyme", "123456"))

        for (method in listOf("gcash", "maya")) {
            assertNull(validatePayoutNumber(method, ""))
            assertNull(validatePayoutNumber(method, "0917 123 4567"))
            assertEquals("Mobile number is 11 digits.", validatePayoutNumber(method, "0917123456"))
            assertEquals("Mobile number is at most 11 digits.", validatePayoutNumber(method, "091712345678"))
            assertEquals("PH mobile numbers start with 09.", validatePayoutNumber(method, "19171234567"))
        }
        assertNull(validatePayoutNumber("gotyme", ""))
        assertEquals("Account number is at least 10 digits.", validatePayoutNumber("gotyme", "123456789"))
        assertNull(validatePayoutNumber("gotyme", "1234 5678 90"))
        assertNull(validatePayoutNumber("gotyme", "1".repeat(16)))
        assertEquals("Account number is at most 16 digits.", validatePayoutNumber("gotyme", "1".repeat(17)))
    }
}
