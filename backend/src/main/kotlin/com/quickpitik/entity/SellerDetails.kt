package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// A photographer's seller-check record (V93), required before new uploads.
// The *Key columns are private storage keys: never put them on a wire DTO.
@Entity
@Table(name = "photographer_seller_details")
class SellerDetails(
    @Id
    @Column(name = "user_id", nullable = false, updatable = false)
    val userId: UUID,

    @Column(name = "legal_name", nullable = false, length = 200)
    var legalName: String,

    @Enumerated(EnumType.STRING)
    @Column(name = "id_type", nullable = false, length = 20)
    var idType: SellerIdType,

    @Column(name = "id_front_key", nullable = false, length = 512)
    var idFrontKey: String,

    @Column(name = "id_back_key", length = 512)
    var idBackKey: String?,

    @Column(name = "selfie_key", nullable = false, length = 512)
    var selfieKey: String,

    @Column(name = "bir_form_key", length = 512)
    var birFormKey: String?,

    @Column(name = "address", nullable = false, length = 500)
    var address: String,

    @Column(name = "phone", nullable = false, length = 30)
    var phone: String,

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 20)
    var status: SellerDetailsStatus,

    @Column(name = "rejection_reason", length = 500)
    var rejectionReason: String? = null,

    @Column(name = "revision", nullable = false)
    var revision: Long,

    @Column(name = "submitted_at", nullable = false)
    var submittedAt: OffsetDateTime,

    @Column(name = "reviewed_at")
    var reviewedAt: OffsetDateTime? = null,

    @Column(name = "reviewed_by")
    var reviewedBy: UUID? = null,

    /** The phone's last 10 digits, for repeat-account matching (V94). */
    @Column(name = "phone_key", nullable = false, length = 10)
    var phoneKey: String,

    // Automatic identity checks (V94; AA-I3 to AA-I5 fill them). Scores use AWS's 0-100 scale.
    @Column(name = "liveness_session_id", length = 128)
    var livenessSessionId: String? = null,

    @Column(name = "liveness_confidence", precision = 5, scale = 2)
    var livenessConfidence: BigDecimal? = null,

    @Column(name = "face_similarity", precision = 5, scale = 2)
    var faceSimilarity: BigDecimal? = null,

    @Column(name = "auto_checked_at")
    var autoCheckedAt: OffsetDateTime? = null,

    @Column(name = "biometric_consent_at")
    var biometricConsentAt: OffsetDateTime? = null,

    @Column(name = "liveness_reference_key", length = 512)
    var livenessReferenceKey: String? = null,

    @Column(name = "id_number_hmac", length = 64)
    var idNumberHmac: String? = null,

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "reason_codes", nullable = false, columnDefinition = "jsonb")
    var reasonCodes: List<SellerReasonCode> = emptyList(),

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "repeat_matches", nullable = false, columnDefinition = "jsonb")
    var repeatMatches: List<RepeatMatch> = emptyList(),
) {
    fun fileKeys(): List<String> = listOfNotNull(idFrontKey, idBackKey, selfieKey, birFormKey, livenessReferenceKey)
}

/** What two seller accounts were found to share. */
enum class RepeatMatchKind { ID_NUMBER, PAYOUT_NUMBER, PHONE }

/** Another account that shares details with this one (stored as JSONB in repeat_matches). */
data class RepeatMatch(val userId: UUID, val matchedOn: List<RepeatMatchKind>)
