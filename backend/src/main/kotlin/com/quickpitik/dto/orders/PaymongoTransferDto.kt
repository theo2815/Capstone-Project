package com.quickpitik.dto.orders

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty

// POST /v2/batch_transfers uses a transfers array, unlike v1 checkout's data envelope.
// https://docs.paymongo.com/docs/money-movement-send-money
data class PaymongoTransferRequest(
    val transfers: List<PaymongoTransferRequestAttributes>,
)

data class PaymongoTransferRequestAttributes(
    // Centavos, matching the checkout boundary (peso × 100).
    val amount: Long,
    val currency: String = "PHP",
    val description: String,
    @JsonProperty("source_account") val sourceAccount: PaymongoTransferAccount,
    @JsonProperty("destination_account") val destinationAccount: PaymongoTransferAccount,
    val provider: String,
    @JsonProperty("callback_url") val callbackUrl: String,
)

data class PaymongoTransferAccount(
    val number: String,
    val name: String,
    val bic: String,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferResponse(
    val transfers: List<PaymongoTransferResult> = emptyList(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferResult(
    val id: String?,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferNode(
    val id: String?,
    val attributes: PaymongoTransferNodeAttributes?,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferNodeAttributes(
    val status: String?,
    @JsonProperty("reference_number") val referenceNumber: String?,
)

// ── Inbound webhook: transfer.outward.successful | failed | returned ──────────
// Envelope mirrors PaymongoWebhookEvent: outer data.id is the EVENT id (evt_…),
// the nested data.data.id is the transfer id (tr_…) we stored on the cycle.
@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferWebhookEvent(
    val data: PaymongoTransferWebhookOuter?,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferWebhookOuter(
    val id: String?, // event id (evt_…) — the dedupe key
    val attributes: PaymongoTransferWebhookAttributes?,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferWebhookAttributes(
    val type: String?, // ⚠ exact strings verify-in-test-mode
    val data: PaymongoTransferNode?, // the transfer node (id + status)
)
