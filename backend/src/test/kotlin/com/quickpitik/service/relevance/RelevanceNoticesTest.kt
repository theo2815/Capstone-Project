package com.quickpitik.service.relevance

import org.junit.jupiter.api.Test
import java.time.LocalDate
import kotlin.test.assertEquals
import kotlin.test.assertFalse

// RV-3.1: each note is pinned to the approved RV-L2 text (V2-T11 to V2-T20, V2-T23; owner, 2026-11-15, "Session 2").
class RelevanceNoticesTest {

    private val by = "DEC 01 · 2026"

    @Test
    fun `the inbox date format is used for the appeal date`() {
        assertEquals("DEC 01 · 2026", RelevanceNotices.date(LocalDate.of(2026, 12, 1)))
    }

    @Test
    fun `the mixed note uses the approved words and the batch's numbers`() {
        assertEquals(RelevanceNote("Some of your photos are off sale",
            "We checked 300 of your 1200 photos in Cebu Run, and 45 of them look blurry, very dark or not of a race. " +
                "Our system is now checking every photo and taking the ones that look like that off sale. Your other photos stay on sale. " +
                "If you think this is wrong, you can appeal from this message in your QuickPitik inbox until DEC 01 · 2026. " +
                "A person at QuickPitik decides every appeal."),
            RelevanceNotices.mixed("Cebu Run", 300, 1200, 45, by))
    }

    @Test
    fun `the junk dump note uses the approved words and the batch's numbers`() {
        assertEquals(RelevanceNote("Your photos in Cebu Run are off sale",
            "We checked 300 of your 1200 photos in Cebu Run, and 250 of them look blurry, very dark or not of a race. " +
                "So our system has taken all your photos in this event off sale. Photos you add to this event are taken off sale too. " +
                "If you think this is wrong, you can appeal from this message in your QuickPitik inbox until DEC 01 · 2026. " +
                "A person at QuickPitik decides every appeal. If you win, we check every photo and put the good ones back on sale. " +
                "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals)."),
            RelevanceNotices.junkDump("Cebu Run", 300, 1200, 250, by))
    }

    @Test
    fun `the wrong event note uses the approved words and the event's date`() {
        assertEquals(RelevanceNote("Your photos in Cebu Run are off sale",
            "Many of your 1200 photos in Cebu Run seem to be from a different event: their camera dates are far from the " +
                "event's date (MAR 04 · 2026), or the words printed in them name a different race. " +
                "So our system has taken all your photos in this event off sale. Photos you add to this event are taken off sale too. " +
                "If your camera's clock was set wrong, or the photos do belong to this event, you can appeal from this message in " +
                "your QuickPitik inbox until DEC 01 · 2026. A person at QuickPitik decides every appeal. " +
                "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals)."),
            RelevanceNotices.wrongEvent("Cebu Run", 1200, "MAR 04 · 2026", by))
    }

    @Test
    fun `the copied note uses the approved words`() {
        assertEquals(RelevanceNote("Photos off sale and payouts on hold",
            "Many of your 1200 photos in Cebu Run match photos that another photographer uploaded. " +
                "So our system has taken all your photos in this event off sale, and your payouts are on hold. " +
                "The hold covers all your payouts, not only this event's sales. Your earnings stay in your balance. " +
                "An admin has been told and will review it. The hold ends when the review is finished or you win an appeal. " +
                "If you took these photos yourself, you can appeal from this message in your QuickPitik inbox until DEC 01 · 2026. " +
                "A person at QuickPitik decides every appeal. " +
                "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals)."),
            RelevanceNotices.copied("Cebu Run", 1200, by))
    }

    @Test
    fun `the copied note does not say copied`() {
        // It takes no other photographer as input, so it names no one, and it does not say "copied from" (PE-L4).
        val note = RelevanceNotices.copied("Cebu Run", 1200, by)
        listOf(note.title, note.body).forEach { assertFalse(it.contains("copied", ignoreCase = true), it) }
    }

    @Test
    fun `the blurry note uses the approved words, one and many`() {
        assertEquals(RelevanceNote("12 photos look blurry",
            "We checked 800 of your photos in Cebu Run. 12 of them look blurry. They are still on sale. " +
                "Review them, and hide any you do not want to sell. You can show them again later."),
            RelevanceNotices.blurry("Cebu Run", 800, 12))
        assertEquals(RelevanceNote("1 photo looks blurry",
            "We checked 800 of your photos in Cebu Run. One of them looks blurry. They are still on sale. " +
                "Review them, and hide any you do not want to sell. You can show them again later."),
            RelevanceNotices.blurry("Cebu Run", 800, 1))
    }

    @Test
    fun `the photos shown again note uses the approved words after a scan and after a re-check`() {
        assertEquals(RelevanceNote("Photos back on sale",
            "We checked every one of your photos in Cebu Run, and fewer look blurry, very dark or not of a race than our first " +
                "check showed. So the photos we took off sale are going back on sale."),
            RelevanceNotices.restoredAfterScan("Cebu Run"))
        assertEquals(RelevanceNote("Photos back on sale",
            "We checked your photos in Cebu Run again after your new uploads. The photos we took off sale are going back on " +
                "sale, except those that still look blurry, very dark or not of a race."),
            RelevanceNotices.restoredAfterRecheck("Cebu Run"))
    }

    @Test
    fun `the appeal received note uses the approved words, copied and not`() {
        val start = "We received your appeal about your photos in Cebu Run. A person at QuickPitik will look at photos from " +
            "the check and decide within 14 days. Until then, "
        assertEquals(RelevanceNote("Appeal received", start + "the photos stay off sale."),
            RelevanceNotices.appealReceived("Cebu Run", copied = false))
        assertEquals(RelevanceNote("Appeal received", start + "the photos stay off sale and your payouts stay on hold."),
            RelevanceNotices.appealReceived("Cebu Run", copied = true))
    }

    @Test
    fun `the appeal won note uses the approved words, whole batch and some photos`() {
        assertEquals(RelevanceNote("Appeal accepted",
            "An admin looked at your appeal about Cebu Run and agreed with you. We are now checking every photo and putting " +
                "the good ones back on sale. Photos that look blurry, very dark or not of a race stay off sale. " +
                "This does not count against you. If your payouts were on hold because of this, the hold has ended."),
            RelevanceNotices.appealWon("Cebu Run", wholeBatch = true))
        assertEquals(RelevanceNote("Appeal accepted",
            "An admin looked at your appeal about Cebu Run and agreed with you. The photos we took off sale are going back " +
                "on sale. This does not count against you."),
            RelevanceNotices.appealWon("Cebu Run", wholeBatch = false))
    }

    @Test
    fun `the appeal lost note uses the approved words and never shows the admin's reason`() {
        val start = "An admin looked at your appeal about Cebu Run and decided the photos stay off sale. " +
            "When all your photos in the event were off sale, this counts as a confirmed case (see the Terms, Appeals). "
        val end = "This is QuickPitik's final decision on the appeal. You can still complain to the National Privacy " +
            "Commission or the Department of Trade and Industry."
        assertEquals(RelevanceNote("Appeal not accepted", start + end), RelevanceNotices.appealLost("Cebu Run", copied = false))
        assertEquals(RelevanceNote("Appeal not accepted",
            start + "Your payouts stay on hold until an admin finishes the review. " + end),
            RelevanceNotices.appealLost("Cebu Run", copied = true))
        // RV-4.5a review: an admin's hide never counts, so the confirmed-case sentence is left out; nothing else changes.
        val opening = "An admin looked at your appeal about Cebu Run and decided the photos stay off sale. "
        assertEquals(RelevanceNote("Appeal not accepted", opening + end), RelevanceNotices.appealLost("Cebu Run", copied = false, counts = false))
        assertEquals(RelevanceNote("Appeal not accepted", opening + "Your payouts stay on hold until an admin finishes the review. " + end),
            RelevanceNotices.appealLost("Cebu Run", copied = true, counts = false))
    }

    @Test
    fun `the second confirmed case note uses the approved words`() {
        assertEquals(RelevanceNote("Second confirmed case",
            "Your photos in Cebu Run were taken off sale, and that decision is now final. This is the second confirmed case " +
                "on your account in 6 months. If there is a third within 6 months, an admin will decide whether to suspend " +
                "your account. Nothing is suspended automatically. To avoid this, upload only your own photos, to the event " +
                "they were taken at, and leave out blurry or very dark shots."),
            RelevanceNotices.secondCase("Cebu Run", windowMonths = 6))
    }

    @Test
    fun `the admin hide note uses the approved words with the new-uploads sentence`() {
        assertEquals(RelevanceNote("Your photos in Cebu Run are off sale",
            "Our automatic check could not tell whether your photos in Cebu Run belong to this event, so a person at " +
                "QuickPitik looked at them and has taken all your photos in this event off sale. Photos you add to this event " +
                "are taken off sale too. If you think this is wrong, you can appeal from this message in your QuickPitik inbox " +
                "until DEC 01 · 2026. A person at QuickPitik decides every appeal. If you win, we check every photo and put the " +
                "good ones back on sale."),
            RelevanceNotices.adminHide("Cebu Run", by))
    }
}
