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

    private fun dispute(
        photoId: String,
        status: String,
        refund: Double? = null,
        note: String = "blurry",
        resolutionNote: String? = null,
    ) = RunnerDisputeDto(
        id = "d-$photoId-$status", photoId = photoId, reason = "quality", note = note, status = status,
        refundAmount = refund, resolutionNote = resolutionNote, openedAt = "2026-09-01T00:00:00Z",
    )

    private fun order(photoIds: List<String>, disputes: List<RunnerDisputeDto>) = OrderDetailDto(
        id = "o1", eventId = "e1", photoIds = photoIds, total = 300.0, paymentMethod = "GCASH", paidAt = null,
        eventName = null, eventSlug = null, status = "PAID", photos = emptyList(), disputes = disputes,
    )
}
