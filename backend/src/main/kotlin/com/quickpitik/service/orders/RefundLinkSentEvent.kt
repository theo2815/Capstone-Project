package com.quickpitik.service.orders

import java.math.BigDecimal
import java.util.UUID

// Published by PaymongoRefundService.sendRefundLink inside its transaction; RefundLinkSentEmailListener
// mails the link after commit. Carries the order, not an address, so a missing order is skipped there.
data class RefundLinkSentEvent(
    val orderId: UUID,
    val url: String,
    val amountPhp: BigDecimal,
    val validityDays: Long,
)
