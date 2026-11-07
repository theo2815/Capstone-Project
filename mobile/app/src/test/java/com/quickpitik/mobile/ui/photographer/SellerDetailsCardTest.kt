package com.quickpitik.mobile.ui.photographer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SellerDetailsCardTest {
    @Test
    fun `missing details show the missing card`() {
        assertEquals(SellerDetailsCard.Missing, sellerDetailsCard("missing", null))
    }

    @Test
    fun `rejected details show the reason`() {
        assertEquals(SellerDetailsCard.Rejected("The ID is expired"), sellerDetailsCard("rejected", "The ID is expired."))
    }

    @Test
    fun `a reason ending in a period loses it so the card shows no double period`() {
        assertEquals(
            SellerDetailsCard.Rejected("the ID photo is a placeholder"),
            sellerDetailsCard("rejected", "  the ID photo is a placeholder.  "),
        )
    }

    @Test
    fun `a reason that is only a period or spaces counts as no reason`() {
        listOf(".", "  ").forEach { assertEquals(it, SellerDetailsCard.Rejected(null), sellerDetailsCard("rejected", it)) }
    }

    @Test
    fun `submitted, approved, unknown and unread statuses show no card`() {
        listOf("submitted", "approved", "something-new", null).forEach { assertNull(it, sellerDetailsCard(it, null)) }
    }
}
