package com.quickpitik.dto.orders

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty

// PayMongo webhook event envelope. We bind only the fields we care about
// and ignore everything else so a future API addition can't break us.
//
// Layout (relevant nesting only):
//
//   { data: {
//       id: "evt_…",
//       type: "event",
//       attributes: {
//         type:     "checkout_session.payment.paid",
//         livemode: false,
//         data: {
//           id:    "cs_…",       // the resource id (checkout session / payment)
//           type:  "checkout_session",
//           attributes: { metadata: { … the map we sent at creation … } }
//         }
//       }
//   }}

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoWebhookEvent(
    val data: PaymongoEventData = PaymongoEventData(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoEventData(
    val id: String = "",
    val type: String = "",
    val attributes: PaymongoEventAttributes = PaymongoEventAttributes(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoEventAttributes(
    val type: String = "",
    val livemode: Boolean = false,
    @JsonProperty("created_at") val createdAt: Long? = null,
    val data: PaymongoResource = PaymongoResource(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoResource(
    val id: String = "",
    val type: String = "",
    val attributes: PaymongoResourceAttributes = PaymongoResourceAttributes(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class PaymongoResourceAttributes(
    val metadata: Map<String, String>? = null,
    val status: String? = null,
)
