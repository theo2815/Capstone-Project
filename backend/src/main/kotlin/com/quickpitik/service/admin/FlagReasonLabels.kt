package com.quickpitik.service.admin

// shortcut: labels copied from the website, upgrade when the website reads reason labels from the backend
// Word for word from website/src/lib/admin-flags.ts FLAG_REASON_LABEL (AX-48).
private val FLAG_REASON_LABEL = mapOf(
    "wrong_runner" to "Wrong runner / face mismatch",
    "low_quality" to "Quality below threshold",
    "inappropriate" to "Inappropriate or NSFW",
    "watermark_bypass" to "Watermark removed or bypassed",
    "duplicate" to "Duplicate of another listing",
    "other" to "Other (see note)",
    "profile_image_flagged" to "Profile image failed the automatic check",
    "profile_image_unchecked" to "Profile image could not be checked",
    "payout_name_changed" to "Payout name changed",
    "repeat_account_match" to "Details match another account",
    "race_check_miss" to "Race failed an automatic check",
    "race_edit_held" to "Race edit waiting for review",
    "possible_duplicate" to "Possible duplicate of another race",
    "same_bib_duplicate" to "Possible duplicate: same bibs as another race",
    "no_proof_hidden" to "Race hidden: no proof it took place",
    "no_proof_unindexed" to "Race has no indexed photos yet",
    "report_threshold_hidden" to "Race hidden after several reports",
    "user_report" to "Reported by a user",
    "runner_removal_held" to "Runner deletion held",
    "batch_relevance_low" to "Uploads may not belong to the event",
)

// AX-48: the reason codes an admin flag search names, by their label ("NSFW") or by the
// code read with spaces ("low quality").
fun flagReasonCodesMatching(query: String): List<String> {
    val q = query.trim().lowercase()
    if (q.isEmpty()) return emptyList()
    return FLAG_REASON_LABEL.filter { (code, label) -> q in label.lowercase() || q in code.replace('_', ' ') }.keys.toList()
}
