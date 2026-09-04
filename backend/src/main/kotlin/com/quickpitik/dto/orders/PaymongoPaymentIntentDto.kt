package com.quickpitik.dto.orders

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty

data class PaymongoPaymentIntentRequest(
    val data: PaymongoPaymentIntentRequestEnvelope,
)

data class PaymongoPaymentIntentRequestEnvelope(
    val attributes: PaymongoPaymentIntentRequestAttributes,
)

data class PaymongoPaymentIntentRequestAttributes(
    val amount: Long,
    val currency: String = "PHP",
    @JsonProperty("payment_method_allowed") val paymentMethodAllowed: List<String> = listOf("qrph"),
    val description: String,
    val metadata: Map<String, String> = emptyMap(),
)

data class PaymongoPaymentMethodRequest(
    val data: PaymongoPaymentMethodRequestEnvelope,
)

data class PaymongoPaymentMethodRequestEnvelope(
    val attributes: PaymongoPaymentMethodRequestAttributes,
)

data class PaymongoPaymentMethodRequestAttributes(
    val type: String = "qrph",
    @JsonProperty("expiry_seconds") val expirySeconds: Int,
    val billing: PaymongoBilling? = null,
)

data class PaymongoPaymentIntentAttachRequest(
    val data: PaymongoPaymentIntentAttachEnvelope,
)

data class PaymongoPaymentIntentAttachEnvelope(
    val attributes: PaymongoPaymentIntentAttachAttributes,
)

data class PaymongoPaymentIntentAttachAttributes(
    @JsonProperty("payment_method") val paymentMethod: String,
    @JsonProperty("client_key") val clientKey: String,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoPaymentMethodResponse(
    val data: PaymongoPaymentMethodResponseEnvelope = PaymongoPaymentMethodResponseEnvelope(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoPaymentMethodResponseEnvelope(
    val id: String = "",
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoPaymentIntentResponse(
    val data: PaymongoPaymentIntentResponseEnvelope = PaymongoPaymentIntentResponseEnvelope(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoPaymentIntentResponseEnvelope(
    val id: String = "",
    val attributes: PaymongoPaymentIntentResponseAttributes = PaymongoPaymentIntentResponseAttributes(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoPaymentIntentResponseAttributes(
    @JsonProperty("client_key") val clientKey: String = "",
    val status: String = "",
    val payments: List<PaymongoPaymentResource> = emptyList(),
    @JsonProperty("next_action") val nextAction: PaymongoNextAction? = null,
    val metadata: Map<String, String>? = null,
    @JsonProperty("updated_at") val updatedAt: Long = 0,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoNextAction(
    val code: PaymongoQrCode? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoQrCode(
    @JsonProperty("image_url") val imageUrl: String = "",
    @JsonProperty("test_url") val testUrl: String? = null,
)
