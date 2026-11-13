package com.quickpitik.dto.admin

import com.quickpitik.entity.SellerDetails
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// E3 row (the admin list). No links: files are opened one record at a time (E4).
data class AdminSellerDetailsRowDto(
    val userId: UUID,
    val email: String,
    val legalName: String,
    val idType: String,
    val status: String,
    val reviewState: String,
    val submittedAt: OffsetDateTime,
    val reviewedAt: OffsetDateTime?,
    val reasonCodes: List<String>,
    val livenessConfidence: BigDecimal?,
    val faceSimilarity: BigDecimal?,
    val autoCheckedAt: OffsetDateTime?,
)

// E4 links, each valid for five minutes; null when that file was not sent.
data class SellerDetailsFilesDto(
    val idFront: String,
    val idBack: String?,
    val selfie: String,
    val birForm: String?,
    val livenessReference: String?,
)

// Another account sharing a detail; email and suspended are read live at view time.
data class RepeatMatchDto(
    val userId: UUID,
    val email: String,
    val matchedOn: List<String>,
    val suspended: Boolean,
)

// E4 to E6. `files` and `linksExpireAt` are set only on E4 (the view writes a decision-log row).
data class AdminSellerDetailsDto(
    val userId: UUID,
    val email: String,
    val status: String,
    val reviewState: String,
    val legalName: String,
    val idType: String,
    val address: String,
    val phone: String,
    val hasIdBack: Boolean,
    val hasBirForm: Boolean,
    val rejectionReason: String?,
    val submittedAt: OffsetDateTime,
    val reviewedAt: OffsetDateTime?,
    val reviewedBy: UUID?,
    val revision: Long,
    val reasonCodes: List<String>,
    val livenessConfidence: BigDecimal?,
    val faceSimilarity: BigDecimal?,
    val autoCheckedAt: OffsetDateTime?,
    val biometricConsentAt: OffsetDateTime?,
    // Only whether a number was given: the raw number is never stored and its hash never leaves the server.
    val hasIdNumber: Boolean,
    val repeatMatches: List<RepeatMatchDto>,
    // Automatic tries that failed a check (R-8); REPEATED_FAILED_CHECKS appears once it reaches the limit.
    val failedAutoChecks: Int,
    val files: SellerDetailsFilesDto? = null,
    val linksExpireAt: OffsetDateTime? = null,
)

fun SellerDetails.toAdminRowDto(email: String) = AdminSellerDetailsRowDto(
    userId = userId,
    email = email,
    legalName = legalName,
    idType = idType.toWire(),
    status = status.toWire(),
    reviewState = status.reviewState(),
    submittedAt = submittedAt,
    reviewedAt = reviewedAt,
    reasonCodes = reasonCodes.map { it.name },
    livenessConfidence = livenessConfidence,
    faceSimilarity = faceSimilarity,
    autoCheckedAt = autoCheckedAt,
)

fun SellerDetails.toAdminDto(
    email: String,
    repeatMatches: List<RepeatMatchDto> = emptyList(),
    files: SellerDetailsFilesDto? = null,
    linksExpireAt: OffsetDateTime? = null,
) =
    AdminSellerDetailsDto(
        userId = userId,
        email = email,
        status = status.toWire(),
        reviewState = status.reviewState(),
        legalName = legalName,
        idType = idType.toWire(),
        address = address,
        phone = phone,
        hasIdBack = idBackKey != null,
        hasBirForm = birFormKey != null,
        rejectionReason = rejectionReason,
        submittedAt = submittedAt,
        reviewedAt = reviewedAt,
        reviewedBy = reviewedBy,
        revision = revision,
        reasonCodes = reasonCodes.map { it.name },
        livenessConfidence = livenessConfidence,
        faceSimilarity = faceSimilarity,
        autoCheckedAt = autoCheckedAt,
        biometricConsentAt = biometricConsentAt,
        hasIdNumber = idNumberHmac != null,
        repeatMatches = repeatMatches,
        failedAutoChecks = failedAutoChecks,
        files = files,
        linksExpireAt = linksExpireAt,
    )

// E5 approve body: the revision the admin saw.
data class SellerDetailsApproveRequest(
    @field:NotNull
    val revision: Long?,
)

// E6 reject body: like RejectUserRequest, plus the revision.
data class SellerDetailsRejectRequest(
    @field:NotNull
    val revision: Long?,
    @field:NotBlank
    @field:Size(max = 500)
    val reason: String,
)

// C16 (AA-I22): photographers the identity notice reaches, and how many mails went out (0 on a dry run).
data class IdentityNoticeResultDto(val recipients: Int, val sent: Int)
