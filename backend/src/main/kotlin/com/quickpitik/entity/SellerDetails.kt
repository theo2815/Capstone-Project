package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
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
) {
    fun fileKeys(): List<String> = listOfNotNull(idFrontKey, idBackKey, selfieKey, birFormKey)
}
