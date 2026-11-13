package com.quickpitik.service.relevance

import com.quickpitik.entity.CaseAction
import com.quickpitik.entity.CaseBand
import com.quickpitik.entity.CaseVerdict

// RV-1.4 (Photo Relevance v2, G13, R8): code has the last word on the numbers. The model's verdict is
// kept only when it fits what the sample, the copy rule and the date rule allow; anything else is
// CANT_TELL with a note saying why (plan section 5.9).

data class ModelAnswer(val verdict: String?, val evidenceUsed: List<String>, val adminReason: String?)

// modelVerdict is the model's answer when it is one of the six values, else null: the raw string is never stored.
data class Settled(val verdict: CaseVerdict, val note: String?, val modelVerdict: CaseVerdict?)

private val wireVerdicts = mapOf(
    "fits" to CaseVerdict.FITS, "mixed" to CaseVerdict.MIXED, "junk_dump" to CaseVerdict.JUNK_DUMP,
    "wrong_event" to CaseVerdict.WRONG_EVENT, "copied" to CaseVerdict.COPIED, "cant_tell" to CaseVerdict.CANT_TELL,
)

// failure: CALL_FAILED, CUT_OFF or BAD_JSON from the caller; null when an answer was parsed.
internal fun settle(
    answer: ModelAnswer?, failure: String?, band: CaseBand, copyLow: Boolean?, dateLow: Boolean?,
    sharedRace: Boolean, wordsPhotos: Int, photos: Int,
): Settled {
    if (failure != null || answer == null) return Settled(CaseVerdict.CANT_TELL, failure ?: "CALL_FAILED", null)
    val model = wireVerdicts[answer.verdict] ?: return Settled(CaseVerdict.CANT_TELL, "UNKNOWN_VALUE", null)
    val note = when {
        model == CaseVerdict.MIXED && band != CaseBand.MIXED -> "CONTRADICTS"
        model == CaseVerdict.JUNK_DUMP && band != CaseBand.OVER -> "CONTRADICTS"
        model == CaseVerdict.FITS && band !in setOf(CaseBand.UNDER, CaseBand.SMALL, CaseBand.UNKNOWN) -> "CONTRADICTS"
        model == CaseVerdict.COPIED && copyLow != true -> "CONTRADICTS"
        model == CaseVerdict.WRONG_EVENT && !(sharedRace && (dateLow == true || wordsPhotos * 2 >= photos)) -> "THIN_EVIDENCE"
        band == CaseBand.SMALL && model !in setOf(CaseVerdict.FITS, CaseVerdict.CANT_TELL) -> "SMALL_BATCH"
        "words_suspicious" in answer.evidenceUsed -> "SUSPICIOUS_WORDS"
        else -> null
    }
    return Settled(if (note == null) model else CaseVerdict.CANT_TELL, note, model)
}

internal fun actionFor(verdict: CaseVerdict): CaseAction = when (verdict) {
    CaseVerdict.MIXED -> CaseAction.HIDE_JUNK
    CaseVerdict.JUNK_DUMP, CaseVerdict.WRONG_EVENT, CaseVerdict.COPIED -> CaseAction.HIDE_ALL
    CaseVerdict.FITS, CaseVerdict.CANT_TELL -> CaseAction.NONE
}
