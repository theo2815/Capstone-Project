package com.quickpitik.service.admin

/**
 * The only list of reasons an admin denies a refund request with (DW-4, plan 5.4). The runner reads
 * [runnerText] in the inbox and on /orders, so the server stores readable text, never the code: old
 * Android builds show the stored text raw. Wording paraphrases refund policy v2 (owner Q4: as written).
 * `bought_together` and `window_closed` equal the RefundBlocker wire codes, so the website can preselect them.
 */
enum class DisputeDenyReason(val code: String, val adminLabel: String, val runnerText: String, val noteRequired: Boolean) {
    WRONG_PHOTO(
        "wrong_photo", "Picked the wrong photo",
        "You picked the wrong photo. Our refund policy doesn't cover a photo chosen by mistake.", false,
    ),
    PREFERENCE(
        "preference", "Doesn't like how it looks",
        "The photo arrived as shown. Our refund policy doesn't cover how a photo looks, such as the angle, the light or your expression.",
        false,
    ),
    CHANGED_MIND(
        "changed_mind", "Changed their mind",
        "The photo arrived fine. Our refund policy doesn't cover changing your mind about a photo.", false,
    ),
    NORMAL_IMPERFECTION(
        "normal_imperfection", "Normal race-day blur, still recognisable",
        "Some blur and grain is normal at a live race, and you can still tell who is in this photo, so our refund policy doesn't cover it.",
        false,
    ),
    DOWNLOAD_WORKS(
        "download_works", "Downloads fine on our side",
        "We checked, and this photo downloads and opens correctly on our side.", false,
    ),
    NO_EXTRA_CHARGE(
        "no_extra_charge", "Only one charge found",
        "Our records show one charge for this photo, so there is no extra charge to refund.", false,
    ),
    BOUGHT_TOGETHER(
        "bought_together", "Bought together with other photos",
        "This photo was bought together with other photos in one payment. PayMongo can only refund a whole payment, " +
            "so we can refund a photo only when it was the only photo in that payment.",
        false,
    ),
    WINDOW_CLOSED(
        "window_closed", "PayMongo's refund window closed",
        "PayMongo can no longer refund this payment because its refund window has closed.", false,
    ),

    // The stored text is the admin's note alone.
    OTHER("other", "Other (write the reason)", "", true),
    ;

    companion object {
        fun fromCode(code: String): DisputeDenyReason? = entries.firstOrNull { it.code == code }
    }
}
