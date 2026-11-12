package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

// Deadlines behind each admin dispute's due time (service/admin/DisputeAttention.kt).
// `reviewBusinessDays`  — policy "review within 3 business days" of filing (Mon-Fri, Manila).
// `refundSendWithin`    — policy "send your refund within 24-48 hours" of approval (QR Ph link).
// `linkExpiringSoon`    — a sent refund link is flagged this long before it expires.
// `refundStuckAfter`    — an in-flight PayMongo refund older than this becomes a refund problem.
@ConfigurationProperties(prefix = "app.disputes")
data class DisputeQueueProperties(
    val reviewBusinessDays: Int = 3,
    val refundSendWithin: Duration = Duration.ofHours(48),
    val linkExpiringSoon: Duration = Duration.ofHours(24),
    val refundStuckAfter: Duration = Duration.ofHours(24),
)
