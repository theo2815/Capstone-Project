package com.quickpitik.dto.orders

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty

// PayMongo Disbursements / Transfers (v2) — request + webhook shapes.
//
// ⚠ VERIFY-IN-TEST-MODE (docs §I). The exact v2 attribute nesting, the transfer
// status strings, and the webhook event type strings are documented ambiguities
// that must be confirmed against a real test-mode account before the adapter is
// enabled. The shapes below follow the documented envelope
// (`{ data: { attributes: {...} } }`) and PayMongo's centavos-as-integer money
// convention, but treat them as provisional. The adapter is disabled by default
// (app.platform.disbursement-provider=manual) precisely so these can't move real
// money until verified.

// ── Create transfer: POST /v2/batch_transfers (single = a one-element batch) ──
data class PaymongoTransferRequest(
    val data: PaymongoTransferRequestData,
)

data class PaymongoTransferRequestData(
    val attributes: PaymongoTransferRequestAttributes,
)

data class PaymongoTransferRequestAttributes(
    // Centavos, matching the checkout boundary (peso × 100).
    val amount: Long,
    val currency: String = "PHP",
    // Our cycle id — the provider-side idempotency + reconciliation key.
    @JsonProperty("reference_number") val referenceNumber: String,
    val description: String,
    val destination: PaymongoTransferDestination,
)

data class PaymongoTransferDestination(
    @JsonProperty("account_number") val accountNumber: String,
    @JsonProperty("account_holder_name") val accountHolderName: String,
    // ⚠ BIC / institution code from GET /v2/transfers/receiving_institutions —
    // NOT the wire method string. Mapped best-effort until verified.
    @JsonProperty("bank_code") val bankCode: String,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoTransferResponse(
    val data: PaymongoTransferNode?,
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
