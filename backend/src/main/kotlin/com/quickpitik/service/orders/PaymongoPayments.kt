package com.quickpitik.service.orders

import com.quickpitik.dto.orders.PaymongoPaymentResource

internal fun List<PaymongoPaymentResource>.paidPaymentId(): String? =
    firstOrNull { it.attributes.status.equals("paid", ignoreCase = true) }?.id?.takeIf(String::isNotBlank)
