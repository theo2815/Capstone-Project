package com.quickpitik.dto.orders

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty

// Wire shape for POST /v1/checkout_sessions (request) and the same
// envelope for the GET retrieve. PayMongo wraps every request and
// response body in `{ data: { attributes: ... } }` — we mirror that
// wrapper exactly and ignore everything in the response we don't need.

data class PaymongoCheckoutSessionRequest(
    val data: PaymongoCheckoutSessionRequestEnvelope,
)

data class PaymongoCheckoutSessionRequestEnvelope(
    val attributes: PaymongoCheckoutSessionAttributes,
)

data class PaymongoCheckoutSessionAttributes(
    @JsonProperty("send_email_receipt") val sendEmailReceipt: Boolean = false,
    @JsonProperty("show_description") val showDescription: Boolean = true,
    @JsonProperty("show_line_items") val showLineItems: Boolean = true,
    @JsonProperty("cancel_url") val cancelUrl: String,
    @JsonProperty("success_url") val successUrl: String,
    @JsonProperty("line_items") val lineItems: List<PaymongoLineItem>,
    @JsonProperty("payment_method_types") val paymentMethodTypes: List<String>,
    val description: String,
    val billing: PaymongoBilling? = null,
    val metadata: Map<String, String> = emptyMap(),
)

data class PaymongoLineItem(
    val name: String,
    val amount: Long, // PHP centavos — peso * 100, integer
    val currency: String = "PHP",
    val quantity: Int = 1,
    val description: String? = null,
)

data class PaymongoBilling(
    val email: String,
    val name: String? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoCheckoutSessionResponse(
    val data: PaymongoCheckoutSessionResponseEnvelope = PaymongoCheckoutSessionResponseEnvelope(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoCheckoutSessionResponseEnvelope(
    val id: String = "",
    val type: String = "checkout_session",
    val attributes: PaymongoCheckoutSessionResponseAttributes = PaymongoCheckoutSessionResponseAttributes(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoCheckoutSessionResponseAttributes(
    @JsonProperty("checkout_url") val checkoutUrl: String = "",
    @JsonProperty("client_key") val clientKey: String = "",
    val status: String = "",
)
