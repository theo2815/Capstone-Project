package com.quickpitik.entity

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException
import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

enum class DisputeStatus(val wire: String) {
    OPEN("open"), RESOLVED("resolved"), DENIED("denied"), ESCALATED("escalated");

    companion object {
        fun fromWire(raw: String): DisputeStatus = entries.first { it.wire == raw }
    }
}

enum class DisputeReason(val wire: String) {
    WRONG_RUNNER("wrong_runner"),
    LOW_QUALITY("low_quality"),
    NOT_RECEIVED("not_received"),
    DUPLICATE_CHARGE("duplicate_charge"),
    OTHER("other");

    companion object {
        fun fromWire(raw: String?): DisputeReason {
            val normalised = raw?.trim()?.lowercase().orEmpty()
            return entries.firstOrNull { it.wire == normalised }
                ?: throw ValidationException(
                    code = ErrorCodes.INVALID_REASON,
                    message = "reason must be one of wrong_runner, low_quality, not_received, duplicate_charge, other",
                    field = "reason",
                )
        }
    }
}

enum class DisputeResolution(val wire: String) {
    REFUND_FULL("refund_full"), REFUND_PARTIAL("refund_partial"), DENY("deny");

    companion object {
        fun fromWire(raw: String): DisputeResolution = entries.first { it.wire == raw }
    }
}

@Entity
@Table(name = "disputes")
class Dispute(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "order_id", nullable = false)
    var orderId: UUID,

    @Column(name = "photo_id", nullable = false)
    var photoId: UUID,

    @Column(name = "runner_id")
    var runnerId: UUID? = null,

    @Column(name = "reason", nullable = false, length = 40)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var reasonWire: String,

    @Column(name = "note", nullable = false, columnDefinition = "TEXT")
    var note: String = "",

    @Column(name = "status", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var statusWire: String = DisputeStatus.OPEN.wire,

    @Column(name = "resolution", length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var resolutionWire: String? = null,

    @Column(name = "refund_amount_php", precision = 12, scale = 2)
    var refundAmountPhp: BigDecimal? = null,

    @Column(name = "opened_at", nullable = false, updatable = false)
    val openedAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "resolved_at")
    var resolvedAt: OffsetDateTime? = null,
) {
    var reason: DisputeReason
        get() = DisputeReason.fromWire(reasonWire)
        set(value) { reasonWire = value.wire }

    var status: DisputeStatus
        get() = DisputeStatus.fromWire(statusWire)
        set(value) { statusWire = value.wire }

    var resolution: DisputeResolution?
        get() = resolutionWire?.let { DisputeResolution.fromWire(it) }
        set(value) { resolutionWire = value?.wire }
}
