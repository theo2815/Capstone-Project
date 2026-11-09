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
}
