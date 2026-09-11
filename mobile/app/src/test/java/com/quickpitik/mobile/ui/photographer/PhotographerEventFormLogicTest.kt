package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.PhotographerEventDetailDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The owned-event form's money-adjacent rules: what the pricing trio becomes,
 * when it counts as changed (a false negative silently drops a price edit, a
 * false positive sends a live event back into review), what create and edit
 * actually send, and the field validators. Plus the coupon editor's two rules.
 */
class PhotographerEventFormLogicTest {
    private fun seed(
        pricingMode: String = "paid",
        pricePerPhoto: Double = 150.0,
        watermarkPolicy: String = "platform",
        pendingChange: Map<String, Any?>? = null,
    ) = PhotographerEventDetailDto(
        id = "ev-1", slug = "cebu", name = "Cebu Run", date = "2026-09-20", location = "Cebu City", state = "live",
        photoCount = 0, salesCount = 0, revenueKept = 0.0, visibility = "public", pricingMode = pricingMode,
        watermarkPolicy = watermarkPolicy, pendingChange = pendingChange, description = "Fun run", organizerName = "Org",
        pricePerPhoto = pricePerPhoto,
    )

    @Test fun `the trio a paid event sends is the typed price on the platform watermark, a free one keeps its own`() {
        assertEquals(PricingTrio("paid", "150", "platform"), pricingTrio("paid", " 150 ", "own"))
        assertEquals(PricingTrio("free", "0", "own"), pricingTrio("free", "999", "own"))
        assertEquals(PricingTrio("free", "0", "none"), pricingTrio("free", "", "none"))
    }

    @Test fun `a trio counts as changed against the pending change when one exists, else the event`() {
        assertTrue(trioChanged(null, PricingTrio("paid", "150", "platform")))
        assertFalse(trioChanged(seed(), PricingTrio("paid", "150", "platform")))
        assertFalse(trioChanged(seed(), PricingTrio("paid", "150.0", "platform")))
        assertTrue(trioChanged(seed(), PricingTrio("paid", "175", "platform")))
        assertTrue(trioChanged(seed(), PricingTrio("free", "0", "own")))
        val pending = seed(pendingChange = mapOf("pricingMode" to "free", "pricePerPhoto" to 0, "watermarkPolicy" to "own"))
        assertFalse(trioChanged(pending, PricingTrio("free", "0", "own")))
        assertTrue(trioChanged(pending, PricingTrio("paid", "150", "platform")))
    }

    @Test fun `create sends every field, edit sends only what changed and the trio only when it changed`() {
        val trio = PricingTrio("paid", "150", "platform")
        val created = ownedEventFormInput(null, false, " Cebu Run ", "2026-09-20", " Cebu City ", "  ", "", "public", trio, "content://cover")
        assertEquals(
            OwnedEventFormInput("Cebu Run", "2026-09-20", "Cebu City", null, null, "public", "paid", "150", "platform", coverUri = "content://cover"),
            created,
        )

        val unchanged = ownedEventFormInput(seed(), true, "Cebu Run", "2026-09-20", "Cebu City", "Org", "Fun run", "public", trio, null)
        assertEquals(OwnedEventFormInput(), unchanged)

        val edited = ownedEventFormInput(seed(), true, "Cebu Night Run", "2026-09-20", "Cebu City", "", "Fun run", "unlisted", PricingTrio("paid", "175", "platform"), null)
        assertEquals("Cebu Night Run", edited.title)
        assertNull(edited.date)
        assertNull(edited.location)
        assertEquals("", edited.organizerName)
        assertNull(edited.description)
        assertEquals("unlisted", edited.visibility)
        assertEquals("paid", edited.pricingMode)
        assertEquals("175", edited.pricePerPhoto)
        assertEquals("platform", edited.watermarkPolicy)
    }

    @Test fun `dates must be ISO and a paid event needs a positive price`() {
        assertTrue(isValidEventDate("2026-09-20"))
        assertFalse(isValidEventDate("20/09/2026"))
        assertFalse(isValidEventDate(""))
        assertTrue(isValidEventPrice("free", ""))
        assertTrue(isValidEventPrice("paid", " 150 "))
        assertFalse(isValidEventPrice("paid", ""))
        assertFalse(isValidEventPrice("paid", "0"))
        assertFalse(isValidEventPrice("paid", "abc"))
    }

    @Test fun `coupon codes are 4-16 upper-case alphanumerics and percents are 1-50 or the 100 giveaway`() {
        assertTrue(isValidCouponCode("SAVE20"))
        assertFalse(isValidCouponCode("save20"))
        assertFalse(isValidCouponCode("ABC"))
        assertFalse(isValidCouponCode("A".repeat(17)))
        assertTrue(isValidCouponPercent(1))
        assertTrue(isValidCouponPercent(50))
        assertTrue(isValidCouponPercent(100))
        assertFalse(isValidCouponPercent(51))
        assertFalse(isValidCouponPercent(0))
        assertFalse(isValidCouponPercent(null))
    }
}
