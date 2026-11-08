package com.quickpitik.dto.photographer

import com.quickpitik.entity.SellerDetails
import org.springframework.web.multipart.MultipartFile
import java.time.OffsetDateTime

// E1/E2 response. Deliberately no links and no storage keys: the photographer
// never needs to see the files again, and only admins open them (SC-1b).
data class SellerDetailsDto(
    /** Today's values (missing / submitted / approved / rejected) for old clients; see SellerDetailsStatus.toWire. */
    val status: String,
    /** The detailed state (missing / manual_review / auto_approved / approved / rejected); new clients read this. */
    val reviewState: String,
    val legalName: String? = null,
    val idType: String? = null,
    val address: String? = null,
    val phone: String? = null,
    val hasIdBack: Boolean? = null,
    val hasBirForm: Boolean? = null,
    val rejectionReason: String? = null,
    val submittedAt: OffsetDateTime? = null,
    val reviewedAt: OffsetDateTime? = null,
) {
    companion object {
        val MISSING = SellerDetailsDto(status = "missing", reviewState = "missing")
    }
}

fun SellerDetails.toDto() = SellerDetailsDto(
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
)

// E2 multipart form. Every part is optional at bind time so SellerDetailsService
// can answer VALIDATION_ERROR naming the missing part.
class SellerDetailsForm(
    var legalName: String? = null,
    var idType: String? = null,
    var address: String? = null,
    var phone: String? = null,
    var idFront: MultipartFile? = null,
    var idBack: MultipartFile? = null,
    var selfie: MultipartFile? = null,
    var birForm: MultipartFile? = null,
)
