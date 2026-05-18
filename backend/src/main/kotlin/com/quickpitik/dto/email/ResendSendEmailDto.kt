package com.quickpitik.dto.email

import com.fasterxml.jackson.annotation.JsonIgnoreProperties

// Resend's POST /emails wire shape. We expose only the fields we use; Resend
// supports much more (attachments, headers, tags, reply_to) — YAGNI for v1.

data class ResendSendEmailRequest(
    val from: String,
    val to: List<String>,
    val subject: String,
    val html: String,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class ResendSendEmailResponse(
    val id: String = "",
)
