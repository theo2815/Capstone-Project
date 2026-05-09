package com.quickpitik.dto.orders

import jakarta.validation.constraints.NotBlank
import java.math.BigDecimal
import java.util.UUID

// Provider-stub payment webhook — real GCash/Maya/card integration is out of scope
// for v1 (Q-008 deferred); this endpoint exists so the contract is locked and so
// admin / ops can manually drive an order from PENDING → PAID via curl during demo.
data class PaymentWebhookRequest(
    val orderId: UUID,
    @field:NotBlank
    val providerRef: String = "",
    val amountPhp: BigDecimal? = null,
    val status: String = "SUCCEEDED",
)

data class PaymentWebhookResponse(
    val orderId: UUID,
    val orderStatus: String,
    val grantsMinted: Int,
)
