package com.quickpitik.service.admin

import com.quickpitik.config.DisputeQueueProperties
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.service.orders.refundCancelsAt
import com.quickpitik.service.orders.refundLinkExpiresAt
import com.quickpitik.service.orders.refundWindowClosesAt
import java.time.DayOfWeek
import java.time.Duration
import java.time.LocalTime
import java.time.OffsetDateTime
import java.time.ZoneId

// What a dispute needs from an admin right now. The one home of "needs an admin":
// the queue views, the badge and the overdue filter all read it. Worked out on read
// from existing timestamps, never stored.
enum class DisputeAttention(val wire: String, val needsAdmin: Boolean) {
    CLOSED("closed", false),
    REFUND_IN_PROGRESS("refund_in_progress", false),
    SEND_LINK("send_link", true),
    AWAITING_CLAIM("awaiting_claim", true),
    LINK_EXPIRED("link_expired", true),
    CHECK_PAYMONGO("check_paymongo", true),
    REFUND_PROBLEM("refund_problem", true),
    ESCALATED("escalated", true),
    REVIEW("review", true),
}

// A QR Ph refund parked for the admin to pay by hand.
internal val PARKED_QRPH_ATTENTIONS = setOf(
    DisputeAttention.SEND_LINK,
    DisputeAttention.AWAITING_CLAIM,
    DisputeAttention.LINK_EXPIRED,
    DisputeAttention.CHECK_PAYMONGO,
)

data class DisputeQueueState(
    val attention: DisputeAttention,
    val dueAt: OffsetDateTime?,
    val overdue: Boolean,
    val linkExpiringSoon: Boolean,
)

// The admin badge: disputes that need an admin, and how many of those are past due.
data class DisputeQueueCounts(val needsAdmin: Long, val overdue: Long)

private val IN_FLIGHT_REFUND_STATUSES = setOf("requesting", "pending", "processing")

// The first matching row wins (plan 5.1).
internal fun queueState(
    dispute: Dispute,
    order: Order?,
    paymongo: PaymongoProperties,
    queue: DisputeQueueProperties,
    now: OffsetDateTime,
): DisputeQueueState {
    val approvedAt = dispute.refundRequestedAt ?: dispute.openedAt
    val refundStatus = dispute.refundStatus
    val (attention, dueAt) = when {
        dispute.status == DisputeStatus.RESOLVED || dispute.status == DisputeStatus.DENIED ||
            dispute.status == DisputeStatus.WITHDRAWN -> DisputeAttention.CLOSED to null
        refundStatus in IN_FLIGHT_REFUND_STATUSES -> {
            val stuckAt = approvedAt.plus(queue.refundStuckAfter)
            if (now.isBefore(stuckAt)) DisputeAttention.REFUND_IN_PROGRESS to stuckAt
            else DisputeAttention.REFUND_PROBLEM to approvedAt
        }
        refundStatus == "manual_review" && order != null && order.paymentMethod == PaymentMethod.QRPH ->
            parkedQrphState(dispute, order, paymongo, queue, approvedAt, now)
        refundStatus == "manual_review" || refundStatus == "failed" -> DisputeAttention.REFUND_PROBLEM to approvedAt
        dispute.status == DisputeStatus.ESCALATED ->
            DisputeAttention.ESCALATED to addBusinessDays(dispute.openedAt, queue.reviewBusinessDays)
        else -> DisputeAttention.REVIEW to addBusinessDays(dispute.openedAt, queue.reviewBusinessDays)
    }
    return DisputeQueueState(
        attention = attention,
        dueAt = dueAt,
        overdue = dueAt != null && !now.isBefore(dueAt),
        linkExpiringSoon = attention == DisputeAttention.AWAITING_CLAIM && dueAt != null &&
            Duration.between(now, dueAt) <= queue.linkExpiringSoon,
    )
}

private fun parkedQrphState(
    dispute: Dispute,
    order: Order,
    paymongo: PaymongoProperties,
    queue: DisputeQueueProperties,
    approvedAt: OffsetDateTime,
    now: OffsetDateTime,
): Pair<DisputeAttention, OffsetDateTime> {
    val linkExpiresAt = paymongo.refundLinkExpiresAt(dispute)
    if (linkExpiresAt == null) {
        // After the window closes the hourly job cancels it and no link can be sent.
        val sendBy = approvedAt.plus(queue.refundSendWithin)
        val windowCloses = paymongo.refundWindowClosesAt(order)
        return DisputeAttention.SEND_LINK to if (windowCloses.isBefore(sendBy)) windowCloses else sendBy
    }
    if (now.isBefore(linkExpiresAt)) return DisputeAttention.AWAITING_CLAIM to linkExpiresAt
    val cancelsAt = paymongo.refundCancelsAt(order, dispute)
    if (now.isBefore(cancelsAt)) return DisputeAttention.LINK_EXPIRED to linkExpiresAt
    return DisputeAttention.CHECK_PAYMONGO to cancelsAt
}

// Mon-Fri in `zone`, keeping the time of day; a weekend start counts from Monday 00:00.
// Holidays are not skipped (owner Q1 default), so a deadline can look due early, never late.
internal fun addBusinessDays(start: OffsetDateTime, days: Int, zone: ZoneId = PH_ZONE): OffsetDateTime {
    var local = start.atZoneSameInstant(zone).toLocalDateTime()
    while (local.dayOfWeek.isWeekend()) local = local.toLocalDate().plusDays(1).atTime(LocalTime.MIDNIGHT)
    repeat(days) {
        do local = local.plusDays(1) while (local.dayOfWeek.isWeekend())
    }
    return local.atZone(zone).toOffsetDateTime()
}

private fun DayOfWeek.isWeekend() = this == DayOfWeek.SATURDAY || this == DayOfWeek.SUNDAY
