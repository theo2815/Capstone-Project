package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

/** Who a liveness session is for. Shared with the runner photo-removal plan (RD-0.1). */
enum class LivenessPurpose { PHOTOGRAPHER_IDENTITY, RUNNER_PHOTO_REMOVAL }

// One AWS Face Liveness session the backend created (V94). Single use: consumedAt is set by a
// conditional claim. eventId is set only for RUNNER_PHOTO_REMOVAL (a CHECK enforces it).
@Entity
@Table(name = "liveness_sessions")
class LivenessSession(
    @Id
    @Column(name = "session_id", nullable = false, updatable = false, length = 128)
    val sessionId: String,

    @Column(name = "user_id", nullable = false, updatable = false)
    val userId: UUID,

    @Enumerated(EnumType.STRING)
    @Column(name = "purpose", nullable = false, updatable = false, length = 30)
    val purpose: LivenessPurpose,

    @Column(name = "event_id", updatable = false)
    val eventId: UUID? = null,

    /** The legal version the consent tick was given under. */
    @Column(name = "consent_version", updatable = false, length = 16)
    val consentVersion: String? = null,

    /** Also the consent time: the tick comes before the session is created. */
    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime,

    @Column(name = "consumed_at")
    var consumedAt: OffsetDateTime? = null,
)
