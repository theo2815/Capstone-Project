package com.quickpitik.dto.admin

// Mirrors website/src/lib/api-admin.ts AdminKpis. Server-computed each call —
// the FE just renders the numbers.
data class AdminKpisDto(
    val pendingVerifications: Long,
    val approvedPhotographers: Long,
    val suspended: Long,
    val liveEvents: Long,
    val decisionsThisWeek: Long,
    val openDisputes: Long,
    val openFlags: Long,
    val pendingPayouts: Long,
    // Photographer-owned events awaiting a decision (V46): new submissions
    // + parked pricing changes.
    val pendingEventRequests: Long = 0,
    // Disputes that need an admin (open, escalated, parked or failed refunds; not refunds in
    // progress), and how many of those are past due (service/admin/DisputeAttention.kt).
    // No defaults: this DTO is built in one place, so the compiler checks the wiring.
    val disputesNeedingAdmin: Long,
    val disputesOverdue: Long,
    // AX-8: queue counts, each equal to its admin list's total. The three relevance tab totals are for the
    // tab bar only (Hidden never reaches 0); relevanceNeedsPerson is the badge. payoutsAwaitingReview is
    // the Payouts "pending review" tab (PENDING only), unlike pendingPayouts (PENDING + SCHEDULED).
    val pendingSellerDetails: Long,
    val removalsNeedingAction: Long,
    val openPayoutReports: Long,
    val relevanceHidden: Long,
    val relevanceCantTell: Long,
    val relevanceAppealsOpen: Long,
    val relevanceNeedsPerson: Long,
    val payoutsAwaitingReview: Long,
)

// Mirrors AdminTrendPoint. Day-bucketed counts for the kpi-trend chart.
data class AdminTrendPointDto(
    val date: String,
    val decisions: Long,
    val disputes: Long,
    val payouts: Long,
    val uploads: Long,
)
