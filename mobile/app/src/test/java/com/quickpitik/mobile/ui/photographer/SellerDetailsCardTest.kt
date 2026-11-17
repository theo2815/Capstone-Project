package com.quickpitik.mobile.ui.photographer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.time.LocalDate

class SellerDetailsCardTest {
    @Test
    fun `auto_approved shows no card`() {
        assertNull(sellerDetailsCard("auto_approved", null))
    }

    @Test
    fun `manual_review paused`() {
        assertEquals(
            SellerDetailsCard.ManualReview(paused = true, dueDate = null),
            sellerDetailsCard("manual_review", null, false, "2026-11-17", today),
        )
    }

    @Test
    fun `manual_review with a date`() {
        assertEquals(
            SellerDetailsCard.ManualReview(paused = false, dueDate = "2026-11-17"),
            sellerDetailsCard("manual_review", null, true, "2026-11-17", today),
        )
    }

    @Test
    fun `a date of today still shows the dated line`() {
        assertEquals(
            SellerDetailsCard.ManualReview(paused = false, dueDate = "2026-11-10"),
            sellerDetailsCard("manual_review", null, true, "2026-11-10", today),
        )
    }

    @Test
    fun `a date in the past hides the dated line`() {
        assertEquals(
            SellerDetailsCard.ManualReview(paused = false, dueDate = null),
            sellerDetailsCard("manual_review", null, true, "2026-11-09", today),
        )
    }

    @Test
    fun `manual review without a date or upload answer uses the meanwhile card`() {
        assertEquals(SellerDetailsCard.ManualReview(paused = false, dueDate = null), sellerDetailsCard("manual_review", null))
    }

    @Test
    fun `an unreadable date hides the dated line`() {
        assertEquals(
            SellerDetailsCard.ManualReview(paused = false, dueDate = null),
            sellerDetailsCard("manual_review", null, true, "invalid", today),
        )
    }

    // R-18: the card's sentence shows "Nov 17, 2026", not the server's "2026-11-17".
    @Test
    fun `the due date reads as a short date`() {
        assertEquals("Nov 17, 2026", sellerDueDateLabel("2026-11-17"))
        assertEquals("Nov 5, 2026", sellerDueDateLabel("2026-11-05"))
    }

    @Test
    fun `an unreadable date is shown as sent`() {
        assertEquals("next Monday", sellerDueDateLabel("next Monday"))
    }

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
        listOf("submitted", "approved", "something_new", null).forEach { assertNull(it, sellerDetailsCard(it, null)) }
    }

    private val today = LocalDate.of(2026, 11, 10)
}
