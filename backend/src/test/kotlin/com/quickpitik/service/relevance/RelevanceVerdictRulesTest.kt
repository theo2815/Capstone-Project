package com.quickpitik.service.relevance

import com.quickpitik.entity.CaseAction
import com.quickpitik.entity.CaseBand
import com.quickpitik.entity.CaseVerdict
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class RelevanceVerdictRulesTest {

    private fun answer(verdict: String?, vararg evidence: String) = ModelAnswer(verdict, evidence.toList(), "reason")

    // Defaults: a clean mixed batch, no copy or date result, no shared race, few words.
    private fun settleIt(
        answer: ModelAnswer?,
        failure: String? = null,
        band: CaseBand = CaseBand.MIXED,
        copyLow: Boolean? = false,
        dateLow: Boolean? = false,
        sharedRace: Boolean = false,
        wordsPhotos: Int = 0,
        photos: Int = 1000,
    ) = settle(answer, failure, band, copyLow, dateLow, sharedRace, wordsPhotos, photos)

    private fun cantTell(note: String, model: CaseVerdict?) = Settled(CaseVerdict.CANT_TELL, note, model)

    @Test fun `a call that did not return settles as can't tell`() {
        assertEquals(cantTell("CALL_FAILED", null), settleIt(null, failure = "CALL_FAILED"))
        assertEquals(cantTell("CALL_FAILED", null), settleIt(null))
    }

    @Test fun `a cut-off answer settles as can't tell`() {
        assertEquals(cantTell("CUT_OFF", null), settleIt(null, failure = "CUT_OFF"))
    }

    @Test fun `an answer that is not the JSON object settles as can't tell`() {
        assertEquals(cantTell("BAD_JSON", null), settleIt(null, failure = "BAD_JSON"))
    }

    @Test fun `an unknown verdict settles as can't tell`() {
        assertEquals(cantTell("UNKNOWN_VALUE", null), settleIt(answer("hide_everything")))
        assertEquals(cantTell("UNKNOWN_VALUE", null), settleIt(answer(null)))
        assertEquals(cantTell("UNKNOWN_VALUE", null), settleIt(answer("MIXED")))
    }

    @Test fun `a 200-character verdict string settles as can't tell with no model verdict`() {
        assertEquals(cantTell("UNKNOWN_VALUE", null), settleIt(answer("mixed" + "x".repeat(195))))
    }

    @Test fun `a verdict outside the band CONTRADICTS`() {
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.MIXED), settleIt(answer("mixed"), band = CaseBand.UNDER))
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.MIXED), settleIt(answer("mixed"), band = CaseBand.UNKNOWN))
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.JUNK_DUMP), settleIt(answer("junk_dump"), band = CaseBand.MIXED))
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.JUNK_DUMP), settleIt(answer("junk_dump"), band = CaseBand.SMALL))
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.FITS), settleIt(answer("fits"), band = CaseBand.MIXED))
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.FITS), settleIt(answer("fits"), band = CaseBand.OVER))
    }

    @Test fun `copied without a low copy result CONTRADICTS`() {
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.COPIED), settleIt(answer("copied"), copyLow = false))
        assertEquals(cantTell("CONTRADICTS", CaseVerdict.COPIED), settleIt(answer("copied"), copyLow = null))
        assertEquals(Settled(CaseVerdict.COPIED, null, CaseVerdict.COPIED), settleIt(answer("copied"), copyLow = true))
    }

    @Test fun `wrong event needs a shared race and a low date or words on half the photos`() {
        val wrong = answer("wrong_event")
        assertEquals(cantTell("THIN_EVIDENCE", CaseVerdict.WRONG_EVENT), settleIt(wrong, sharedRace = false, dateLow = true))
        assertEquals(cantTell("THIN_EVIDENCE", CaseVerdict.WRONG_EVENT), settleIt(wrong, sharedRace = true, dateLow = null, wordsPhotos = 499))
        val kept = Settled(CaseVerdict.WRONG_EVENT, null, CaseVerdict.WRONG_EVENT)
        assertEquals(kept, settleIt(wrong, sharedRace = true, dateLow = true))
        assertEquals(kept, settleIt(wrong, sharedRace = true, dateLow = false, wordsPhotos = 500))
    }

    @Test fun `a small batch allows only fits and can't tell`() {
        assertEquals(cantTell("SMALL_BATCH", CaseVerdict.COPIED), settleIt(answer("copied"), band = CaseBand.SMALL, copyLow = true))
        assertEquals(cantTell("SMALL_BATCH", CaseVerdict.WRONG_EVENT),
            settleIt(answer("wrong_event"), band = CaseBand.SMALL, sharedRace = true, dateLow = true))
        assertEquals(Settled(CaseVerdict.FITS, null, CaseVerdict.FITS), settleIt(answer("fits"), band = CaseBand.SMALL))
        assertEquals(Settled(CaseVerdict.CANT_TELL, null, CaseVerdict.CANT_TELL), settleIt(answer("cant_tell"), band = CaseBand.SMALL))
    }

    @Test fun `suspicious words settle as can't tell`() {
        assertEquals(cantTell("SUSPICIOUS_WORDS", CaseVerdict.MIXED), settleIt(answer("mixed", "junk_share", "words_suspicious")))
        assertEquals(cantTell("SUSPICIOUS_WORDS", CaseVerdict.FITS), settleIt(answer("fits", "words_suspicious"), band = CaseBand.UNDER))
    }

    @Test fun `a clean answer inside the band is kept`() {
        assertEquals(Settled(CaseVerdict.MIXED, null, CaseVerdict.MIXED), settleIt(answer("mixed", "junk_share"), band = CaseBand.MIXED))
        assertEquals(Settled(CaseVerdict.JUNK_DUMP, null, CaseVerdict.JUNK_DUMP), settleIt(answer("junk_dump"), band = CaseBand.OVER))
        assertEquals(Settled(CaseVerdict.FITS, null, CaseVerdict.FITS), settleIt(answer("fits"), band = CaseBand.UNDER))
        assertEquals(Settled(CaseVerdict.FITS, null, CaseVerdict.FITS), settleIt(answer("fits"), band = CaseBand.UNKNOWN))
        assertEquals(Settled(CaseVerdict.CANT_TELL, null, CaseVerdict.CANT_TELL), settleIt(answer("cant_tell"), band = CaseBand.OVER))
    }

    @Test fun `mixed hides junk, the three whole-batch verdicts hide all, fits and can't tell hide nothing`() {
        assertEquals(CaseAction.HIDE_JUNK, actionFor(CaseVerdict.MIXED))
        assertEquals(CaseAction.HIDE_ALL, actionFor(CaseVerdict.JUNK_DUMP))
        assertEquals(CaseAction.HIDE_ALL, actionFor(CaseVerdict.WRONG_EVENT))
        assertEquals(CaseAction.HIDE_ALL, actionFor(CaseVerdict.COPIED))
        assertEquals(CaseAction.NONE, actionFor(CaseVerdict.FITS))
        assertEquals(CaseAction.NONE, actionFor(CaseVerdict.CANT_TELL))
    }
}
