package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.OffsetDateTime
import java.util.UUID

// Photographer-filed report against a cycle. Partial unique index in V9
// (`uq_payout_reports_open_per_cycle`) enforces at-most-one OPEN per (cycle,
// photographer) so the FE's "File a report" vs "Track your report" button
// state can't drift from the DB under concurrent posts.
@Entity
@Table(name = "payout_reports")
class PayoutReport(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "payout_id", nullable = false, updatable = false, length = 80)
    val payoutId: String,

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    @Column(name = "reason", nullable = false, length = 40)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var reasonWire: String,

    @Column(name = "note", nullable = false, columnDefinition = "TEXT")
    var note: String = "",

    @Column(name = "status", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var statusWire: String = PayoutReportStatus.OPEN.wire,

    @Column(name = "opened_at", nullable = false, updatable = false)
    val openedAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "acknowledged_at")
    var acknowledgedAt: OffsetDateTime? = null,

    @Column(name = "acknowledge_reply", columnDefinition = "TEXT")
    var acknowledgeReply: String? = null,

    @Column(name = "resolved_at")
    var resolvedAt: OffsetDateTime? = null,

    @Column(name = "resolution_note", columnDefinition = "TEXT")
    var resolutionNote: String? = null,
) {
    var reason: PayoutReportReason
        get() = PayoutReportReason.fromWire(reasonWire)
            ?: throw IllegalStateException("Unknown payout report reason on $id: $reasonWire")
        set(value) {
            reasonWire = value.wire
        }

    var status: PayoutReportStatus
        get() = PayoutReportStatus.fromWire(statusWire)
        set(value) {
            statusWire = value.wire
        }
}
