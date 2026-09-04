package com.quickpitik.dto.orders

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty

data class PaymongoRefundRequest(
    val data: PaymongoRefundRequestEnvelope,
)

data class PaymongoRefundRequestEnvelope(
    val attributes: PaymongoRefundRequestAttributes,
)

data class PaymongoRefundRequestAttributes(
    val amount: Long,
    @JsonProperty("payment_id") val paymentId: String,
    val reason: String = "requested_by_customer",
    val notes: String? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoRefundResponse(
    val data: PaymongoRefundResource = PaymongoRefundResource(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoRefundResource(
    val id: String = "",
    val type: String = "refund",
    val attributes: PaymongoRefundAttributes = PaymongoRefundAttributes(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoRefundAttributes(
    val amount: Long = 0,
    @JsonProperty("payment_id") val paymentId: String = "",
    val status: String = "",
)
