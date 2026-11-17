package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotographerPayoutDto
import com.quickpitik.mobile.ui.theme.StatusTone
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The hold decisions are pure functions beside their screens (mobile/AGENTS.md:157). */
class PayoutHoldTest {
    private fun balance(hold: String? = null, open: PhotographerPayoutDto? = null, unpaid: Double = 500.0) =
        PayoutBalanceDto(
            unpaidBalance = unpaid, minimum = 100.0, hasOpenRequest = open != null, openRequest = open,
            payoutHoldReason = hold,
        )

    private val pending = PhotographerPayoutDto(
        id = "py-1", weekOf = "2026-09-07", amount = 500.0, status = "pending", settledAt = null,
        method = "gcash", reference = null, holdReason = null,
    )

    @Test fun `no reason is no hold, and any reason at all is a hold`() {
        assertFalse(payoutHoldActive(balance(null)))
        assertTrue(payoutHoldActive(balance("UPLOAD_REVIEW")))
        assertTrue(payoutHoldActive(balance("SOMETHING_NEW")))
    }

    @Test fun `the Home billing line says on hold, ahead of an open request and a ready balance`() {
        assertEquals("Payouts on hold" to StatusTone.Warning, billingGlanceLine(balance("UPLOAD_REVIEW")))
        assertEquals("Payouts on hold" to StatusTone.Warning, billingGlanceLine(balance("UPLOAD_REVIEW", pending)))
        assertEquals("Ready to request" to StatusTone.Approved, billingGlanceLine(balance()))
        assertEquals("Payout pending" to StatusTone.Warning, billingGlanceLine(balance(open = pending)))
        assertEquals("Balance building" to StatusTone.Neutral, billingGlanceLine(balance(unpaid = 10.0)))
    }
}
