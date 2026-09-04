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
)

// Mirrors AdminTrendPoint. Day-bucketed counts for the kpi-trend chart.
data class AdminTrendPointDto(
    val date: String,
    val decisions: Long,
    val disputes: Long,
    val payouts: Long,
)
