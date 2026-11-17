package com.quickpitik.service.admin

import com.quickpitik.config.DisputeQueueProperties
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Order
import org.junit.jupiter.api.Test
import java.math.BigDecimal
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DisputeAttentionTest {
    private val paymongo = PaymongoProperties()
    private val queue = DisputeQueueProperties()
    private val now = OffsetDateTime.parse("2026-11-09T04:00:00Z")

    // ─── addBusinessDays ──────────────────────────────────────────────────

    @Test
    fun `three business days from a Friday afternoon is the next Wednesday afternoon`() {
        assertEquals(
            OffsetDateTime.parse("2026-11-11T15:00+08:00"),
            addBusinessDays(OffsetDateTime.parse("2026-11-06T15:00+08:00"), 3),
        )
    }

    @Test
    fun `a request filed on Saturday starts counting on Monday`() {
        assertEquals(
            OffsetDateTime.parse("2026-11-12T00:00+08:00"),
            addBusinessDays(OffsetDateTime.parse("2026-11-07T10:00+08:00"), 3),
        )
    }

    @Test
    fun `the count runs on Manila time, not UTC`() {
        // Friday 17:00 UTC is Saturday 01:00 in Manila.
        assertEquals(
            OffsetDateTime.parse("2026-11-12T00:00+08:00"),
            addBusinessDays(OffsetDateTime.parse("2026-11-06T17:00Z"), 3),
        )
    }

    // ─── queueState, one test per row of plan 5.1 ─────────────────────────

    @Test
    fun `a resolved, denied or withdrawn dispute is closed with no due time`() {
        for (status in listOf(DisputeStatus.RESOLVED, DisputeStatus.DENIED, DisputeStatus.WITHDRAWN)) {
            val d = dispute(status = status).apply { refundStatus = "succeeded" }
            val s = queueState(d, qrphOrder(), paymongo, queue, now)
            assertEquals(DisputeAttention.CLOSED, s.attention, status.wire)
            assertNull(s.dueAt)
            assertFalse(s.overdue)
            assertFalse(s.attention.needsAdmin)
        }
    }

    @Test
    fun `a refund PayMongo is still working on is in progress until refundStuckAfter`() {
        val requested = now.minusHours(1)
        val d = dispute().apply { refundStatus = "pending"; refundRequestedAt = requested }
        val s = queueState(d, cardOrder(), paymongo, queue, now)
        assertEquals(DisputeAttention.REFUND_IN_PROGRESS, s.attention)
        assertEquals(requested.plusHours(24), s.dueAt)
        assertFalse(s.overdue)
        assertFalse(s.attention.needsAdmin)
    }

    @Test
    fun `a card refund still pending 24 hours after approval becomes a refund problem`() {
        val requested = now.minusHours(24)
        val d = dispute().apply { refundStatus = "pending"; refundRequestedAt = requested }
        val s = queueState(d, cardOrder(), paymongo, queue, now)
        assertEquals(DisputeAttention.REFUND_PROBLEM, s.attention)
        assertEquals(requested, s.dueAt)
        assertTrue(s.overdue)
        assertTrue(s.attention.needsAdmin)
    }

    @Test
    fun `send_link is due exactly 48 hours after approval`() {
        val requested = now.minusHours(10)
        val d = dispute().apply { refundStatus = "manual_review"; refundRequestedAt = requested }
        val order = qrphOrder(paidAt = requested.minusDays(1))
        val due = requested.plusHours(48)

        val before = queueState(d, order, paymongo, queue, due.minusSeconds(1))
        assertEquals(DisputeAttention.SEND_LINK, before.attention)
        assertEquals(due, before.dueAt)
        assertFalse(before.overdue)
        assertTrue(before.attention.needsAdmin)

        assertTrue(queueState(d, order, paymongo, queue, due).overdue)
        assertTrue(queueState(d, order, paymongo, queue, due.plusSeconds(1)).overdue)
    }

    @Test
    fun `a send_link approved 10 hours before PayMongo's window closes is due at the window close`() {
        val requested = now.minusHours(1)
        // The 30-day window closes 10 hours after the approval.
        val order = qrphOrder(paidAt = requested.plusHours(10).minusDays(30))
        val d = dispute().apply { refundStatus = "manual_review"; refundRequestedAt = requested }
        val s = queueState(d, order, paymongo, queue, now)
        assertEquals(DisputeAttention.SEND_LINK, s.attention)
        assertEquals(requested.plusHours(10), s.dueAt)
    }

    @Test
    fun `awaiting_claim is due when the link expires and flags it 24 hours ahead`() {
        val sent = now.minusDays(1)
        val expires = sent.plusDays(3)
        val d = dispute().apply {
            refundStatus = "manual_review"; refundRequestedAt = sent.minusHours(2); refundLinkSentAt = sent
        }
        val order = qrphOrder(paidAt = sent.minusDays(2))

        val soon = queueState(d, order, paymongo, queue, expires.minusHours(23))
        assertEquals(DisputeAttention.AWAITING_CLAIM, soon.attention)
        assertEquals(expires, soon.dueAt)
        assertFalse(soon.overdue)
        assertTrue(soon.linkExpiringSoon)
        assertTrue(soon.attention.needsAdmin)

        assertFalse(queueState(d, order, paymongo, queue, expires.minusHours(25)).linkExpiringSoon)
    }

    @Test
    fun `an expired link before the cancel time is link_expired, due at the expiry`() {
        val sent = now.minusDays(4)
        val d = dispute().apply {
            refundStatus = "manual_review"; refundRequestedAt = sent.minusHours(2); refundLinkSentAt = sent
        }
        // Window still open for weeks, so the cancel time is the window close.
        val s = queueState(d, qrphOrder(paidAt = sent.minusDays(1)), paymongo, queue, now)
        assertEquals(DisputeAttention.LINK_EXPIRED, s.attention)
        assertEquals(sent.plusDays(3), s.dueAt)
        assertTrue(s.overdue)
        assertFalse(s.linkExpiringSoon)
    }

    @Test
    fun `a sent link past the cancel time is check_paymongo, due at the cancel time`() {
        val paid = now.minusDays(40)
        val sent = paid.plusDays(29)
        val d = dispute().apply {
            refundStatus = "manual_review"; refundRequestedAt = sent.minusHours(2); refundLinkSentAt = sent
        }
        val s = queueState(d, qrphOrder(paidAt = paid), paymongo, queue, now)
        assertEquals(DisputeAttention.CHECK_PAYMONGO, s.attention)
        assertEquals(sent.plusDays(3), s.dueAt) // link expiry is later than the window close
        assertTrue(s.overdue)
    }

    @Test
    fun `a legacy card manual review and a failed refund are refund problems due from approval`() {
        val requested = now.minusHours(2)
        for (status in listOf("manual_review", "failed")) {
            val d = dispute().apply { refundStatus = status; refundRequestedAt = requested }
            val s = queueState(d, cardOrder(), paymongo, queue, now)
            assertEquals(DisputeAttention.REFUND_PROBLEM, s.attention, status)
            assertEquals(requested, s.dueAt)
            assertTrue(s.overdue)
        }
    }

    @Test
    fun `a refund problem with no approval time counts from when the dispute opened`() {
        val opened = now.minusDays(1)
        val s = queueState(dispute(openedAt = opened).apply { refundStatus = "failed" }, cardOrder(), paymongo, queue, now)
        assertEquals(opened, s.dueAt)
    }

    @Test
    fun `a failed refund that is tried again becomes refund_in_progress and leaves the needs-you count`() {
        val d = dispute().apply { refundStatus = "failed"; refundRequestedAt = now.minusDays(1) }
        assertTrue(queueState(d, cardOrder(), paymongo, queue, now).attention.needsAdmin)

        d.refundStatus = "requesting"
        d.refundRequestedAt = now
        val retried = queueState(d, cardOrder(), paymongo, queue, now)
        assertEquals(DisputeAttention.REFUND_IN_PROGRESS, retried.attention)
        assertFalse(retried.attention.needsAdmin)
    }

    @Test
    fun `an escalated dispute is due three business days after it opened`() {
        val opened = OffsetDateTime.parse("2026-11-06T15:00+08:00")
        val s = queueState(dispute(status = DisputeStatus.ESCALATED, openedAt = opened), cardOrder(), paymongo, queue, now)
        assertEquals(DisputeAttention.ESCALATED, s.attention)
        assertEquals(OffsetDateTime.parse("2026-11-11T15:00+08:00"), s.dueAt)
        assertFalse(s.overdue)
        assertTrue(s.attention.needsAdmin)
    }

    @Test
    fun `an open dispute needs review within three business days and is overdue from that instant`() {
        val opened = OffsetDateTime.parse("2026-11-06T15:00+08:00")
        val due = OffsetDateTime.parse("2026-11-11T15:00+08:00")
        val d = dispute(openedAt = opened)
        val s = queueState(d, null, paymongo, queue, due.minusSeconds(1))
        assertEquals(DisputeAttention.REVIEW, s.attention)
        assertEquals(due, s.dueAt)
        assertFalse(s.overdue)
        assertTrue(queueState(d, null, paymongo, queue, due).overdue)
    }

    @Test
    fun `the deadlines come from config`() {
        val custom = DisputeQueueProperties(reviewBusinessDays = 1, refundStuckAfter = Duration.ofHours(2))
        val opened = OffsetDateTime.parse("2026-11-06T15:00+08:00")
        assertEquals(
            OffsetDateTime.parse("2026-11-09T15:00+08:00"),
            queueState(dispute(openedAt = opened), null, paymongo, custom, now).dueAt,
        )
        val d = dispute().apply { refundStatus = "pending"; refundRequestedAt = now.minusHours(2) }
        assertEquals(DisputeAttention.REFUND_PROBLEM, queueState(d, cardOrder(), paymongo, custom, now).attention)
    }

    @Test
    fun `wire values are the lower-case names`() {
        assertEquals(
            listOf(
                "closed", "refund_in_progress", "send_link", "awaiting_claim", "link_expired",
                "check_paymongo", "refund_problem", "escalated", "review",
            ),
            DisputeAttention.entries.map { it.wire },
        )
    }

    private fun dispute(status: DisputeStatus = DisputeStatus.OPEN, openedAt: OffsetDateTime = now.minusDays(1)) =
        Dispute(
            orderId = UUID.randomUUID(),
            photoId = UUID.randomUUID(),
            reasonWire = "other",
            statusWire = status.wire,
            openedAt = openedAt,
        )

    private fun qrphOrder(paidAt: OffsetDateTime = now.minusDays(2)) = Order(
        eventId = UUID.randomUUID(),
        recipientEmail = "runner@example.test",
        paymentMethodWire = "qrph",
        totalPhp = BigDecimal("150.00"),
        paidAt = paidAt,
    )

    private fun cardOrder() = Order(
        eventId = UUID.randomUUID(),
        recipientEmail = "runner@example.test",
        paymentMethodWire = "card",
        totalPhp = BigDecimal("150.00"),
        paidAt = now.minusDays(2),
    )
}
