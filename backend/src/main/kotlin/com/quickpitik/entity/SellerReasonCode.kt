package com.quickpitik.entity

// Why an automatic identity check passed or sent a seller to an admin. Stored as a JSONB list on
// photographer_seller_details.reason_codes; admins see them, photographers never do. `[]` with no
// auto_checked_at = sent before automatic checks existed.
enum class SellerReasonCode {
    PASSED,
    LIVENESS_LOW,
    FACE_MISMATCH,
    NO_FACE_ON_ID,
    AI_DISABLED,
    AI_ERROR,
    DECLINED,
    ID_NUMBER_MISMATCH,
    ID_NUMBER_UNREADABLE,
    PAYOUT_NAME_MISMATCH,
    REPEAT_ACCOUNT_MATCH,
    PREVIOUSLY_REJECTED,
    // The number was found on the card, but its ID type has no format set: an admin checks it (R-1).
    ID_FORMAT_UNCHECKED,
    // A word of the typed legal name is not printed on the ID front: an admin compares them (R-2).
    ID_NAME_MISMATCH,
    // The record already holds app.identity.max-failed-auto-checks failed tries: an admin looks, even at a pass (R-8).
    REPEATED_FAILED_CHECKS,
}
