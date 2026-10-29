package com.quickpitik.mobile.ui.runner

import com.quickpitik.mobile.data.remote.OrderDetailDto
import com.quickpitik.mobile.data.remote.RunnerDisputeDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The refund rules ported from the website's getOrderRefundStatus: which photos
 * a runner may still dispute, what the order-row chip says, and when a new
 * request or a cancellation is allowed. Money-adjacent, pure, and previously
 * unasserted.
 */
class RefundComponentsTest {
    @Test fun `denied and withdrawn disputes free the photo again, everything else locks it`() {
        val order = order(
            listOf("a", "b", "c", "d", "e"),
            listOf(
                dispute("a", "open"), dispute("b", "escalated"), dispute("c", "resolved", refund = 100.0),
                dispute("d", "denied"), dispute("e", "withdrawn"),
            ),
        )
        assertEquals(listOf("d", "e"), disputablePhotoIds(order))
    }

    @Test fun `the rollup kind mirrors the website`() {
        assertEquals("none", computeRefundRollup(3, emptyList()).kind)
        assertEquals("none", computeRefundRollup(3, listOf(dispute("a", "withdrawn"))).kind)
        assertEquals("pending", computeRefundRollup(1, listOf(dispute("a", "open"))).kind)
        assertEquals("partial", computeRefundRollup(3, listOf(dispute("a", "open"))).kind)
        assertEquals("approved", computeRefundRollup(1, listOf(dispute("a", "resolved", refund = 100.0))).kind)
        assertEquals("rejected", computeRefundRollup(1, listOf(dispute("a", "denied"))).kind)
        assertEquals(
            "partial",
            computeRefundRollup(2, listOf(dispute("a", "resolved", refund = 100.0), dispute("b", "denied"))).kind,
        )
        // Resolved with no amount is not an approval; it falls through to the mixed bucket.
        assertEquals("partial", computeRefundRollup(1, listOf(dispute("a", "resolved"))).kind)
    }

    @Test fun `the rollup sums approved refunds and prefers the admin's note over the runner's`() {
        val rollup = computeRefundRollup(
            4,
            listOf(
                dispute("a", "resolved", refund = 100.0),
                dispute("b", "resolved", refund = 50.0),
                dispute("c", "denied", note = "mine", resolutionNote = "Photo is sharp."),
                dispute("d", "withdrawn"),
            ),
        )
        assertEquals(150.0, rollup.refundAmount, 0.0)
        assertEquals(2, rollup.approvedCount)
        assertEquals(1, rollup.rejectedCount)
        assertEquals(0, rollup.pendingCount)
        assertEquals(3, rollup.totalDisputed)
        assertEquals("Photo is sharp.", rollup.rejectedNote)
        assertEquals("mine", computeRefundRollup(1, listOf(dispute("c", "denied", note = "mine"))).rejectedNote)
    }

    @Test fun `a new request needs no live dispute and a free photo, only open disputes cancel`() {
        assertTrue(canRequestRefund(order(listOf("a"), emptyList())))
        assertTrue(canRequestRefund(order(listOf("a"), listOf(dispute("a", "denied")))))
        assertFalse(canRequestRefund(order(listOf("a"), listOf(dispute("a", "open")))))
        assertFalse(canRequestRefund(order(listOf("a"), listOf(dispute("a", "resolved", refund = 10.0)))))
        assertFalse(canRequestRefund(order(emptyList(), emptyList())))
        val mixed = order(listOf("a", "b"), listOf(dispute("b", "escalated"), dispute("a", "open")))
        assertEquals("a", cancellableDispute(mixed)?.photoId)
        assertNull(cancellableDispute(order(listOf("a"), listOf(dispute("a", "escalated")))))
    }

    @Test fun `the label says what is happening with a manual refund`() {
        val waiting = "Refund approved, waiting for PayMongo refund link"
        assertEquals(waiting, refundStatusLabel(dispute("a", "escalated", refundStatus = "manual_review")))
        assertEquals(waiting, refundStatusLabel(dispute("a", "escalated", refundStatus = "manual_review", link = "  ")))
        assertEquals(
            "Refund link sent",
            refundStatusLabel(dispute("a", "escalated", refundStatus = "manual_review", link = "https://pm.link/x")),
        )
        assertEquals(
            "Refund cancelled: the refund window closed",
            refundStatusLabel(dispute("a", "denied", refundStatus = "cancelled")),
        )
        assertNull(refundStatusLabel(dispute("a", "escalated")))
        assertNull(refundStatusLabel(dispute("a", "resolved", refund = 10.0, refundStatus = "succeeded")))
        assertNull(refundStatusLabel(dispute("a", "denied")))
    }

    @Test fun `the claim link shows only while the refund is waiting to be claimed`() {
        val url = "https://pm.link/x"
        assertEquals(url, refundClaimLink(dispute("a", "escalated", refundStatus = "manual_review", link = url)))
        assertNull(refundClaimLink(dispute("a", "escalated", refundStatus = "manual_review")))
        assertNull(refundClaimLink(dispute("a", "escalated", refundStatus = "manual_review", link = " ")))
        assertNull(refundClaimLink(dispute("a", "resolved", refund = 10.0, refundStatus = "succeeded", link = url)))
        assertNull(refundClaimLink(dispute("a", "denied", refundStatus = "cancelled", link = url)))
    }

    @Test fun `the rollup label is the first one from a dispute that is not withdrawn`() {
        val waiting = dispute("a", "escalated", refundStatus = "manual_review")
        val sent = dispute("b", "escalated", refundStatus = "manual_review", link = "https://pm.link/x")
        val withdrawn = dispute("c", "withdrawn", refundStatus = "manual_review")
        assertEquals(
            "Refund link sent",
            computeRefundRollup(3, listOf(dispute("x", "open"), sent, waiting)).label,
        )
        assertEquals(
            "Refund approved, waiting for PayMongo refund link",
            computeRefundRollup(3, listOf(withdrawn, waiting, sent)).label,
        )
        assertNull(computeRefundRollup(1, listOf(withdrawn)).label)
        assertNull(computeRefundRollup(1, listOf(dispute("a", "open"))).label)
        assertEquals("pending", computeRefundRollup(1, listOf(waiting)).kind)
    }

    @Test fun `the refund policy copy matches the website`() {
        val bullets = REFUND_POLICY_BULLETS.toMap()
        assertEquals(
            "We review every request within 3 business days. You'll see the status update on this receipt.",
            bullets["Review time"],
        )
        assertEquals(
            "Approved refunds come as a PayMongo refund link, shown on this receipt with its claim-by date and sent to your e-mail. Open it to choose where the money goes.",
            bullets["Where the money goes"],
        )
    }

    private fun dispute(
        photoId: String,
        status: String,
        refund: Double? = null,
        note: String = "blurry",
        resolutionNote: String? = null,
        refundStatus: String? = null,
        link: String? = null,
    ) = RunnerDisputeDto(
        id = "d-$photoId-$status", photoId = photoId, reason = "quality", note = note, status = status,
        refundAmount = refund, resolutionNote = resolutionNote, openedAt = "2026-09-01T00:00:00Z",
        refundStatus = refundStatus, refundLinkUrl = link,
    )

    private fun order(photoIds: List<String>, disputes: List<RunnerDisputeDto>) = OrderDetailDto(
        id = "o1", eventId = "e1", photoIds = photoIds, total = 300.0, paymentMethod = "GCASH", paidAt = null,
        eventName = null, eventSlug = null, status = "PAID", photos = emptyList(), disputes = disputes,
    )
}
