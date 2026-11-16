package com.quickpitik.service.relevance

import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale

data class RelevanceNote(val title: String, val body: String)

// RV-3.1 (A.5, G12): every note a photographer reads about Photo Relevance v2, copied character for character from the
// approved RV-L2 texts (V2-T11 to V2-T20 and V2-T23; owner, 2026-11-15, "Session 2"). Code fills in the values only.
// Claude's reason is never an input. The notify step sends V2-T11 to V2-T16 and V2-T23; appeals (RV-3.2) and the
// ladder (RV-4.1) call the others.
object RelevanceNotices {
    // The date format the inbox already uses (website lib/format.ts formatLongDate): "DEC 01 · 2026".
    private val DATE: DateTimeFormatter = DateTimeFormatter.ofPattern("MMM dd · yyyy", Locale.ENGLISH)

    fun date(day: LocalDate): String = day.format(DATE).uppercase(Locale.ENGLISH)

    // V2-T11 (the bracketed link sentence left out, "Session 2" choice 5).
    fun mixed(eventName: String, sampleSize: Int, batchPhotos: Int, junkCount: Int, appealBy: String) = RelevanceNote(
        "Some of your photos are off sale",
        "We checked $sampleSize of your $batchPhotos photos in $eventName, and $junkCount of them look blurry, very dark or not of a race. " +
            "Our system is now checking every photo and taking the ones that look like that off sale. Your other photos stay on sale. " +
            "If you think this is wrong, you can appeal from this message in your QuickPitik inbox until $appealBy. " +
            "A person at QuickPitik decides every appeal.",
    )

    // V2-T12.
    fun junkDump(eventName: String, sampleSize: Int, batchPhotos: Int, junkCount: Int, appealBy: String) = RelevanceNote(
        "Your photos in $eventName are off sale",
        "We checked $sampleSize of your $batchPhotos photos in $eventName, and $junkCount of them look blurry, very dark or not of a race. " +
            "So our system has taken all your photos in this event off sale. Photos you add to this event are taken off sale too. " +
            "If you think this is wrong, you can appeal from this message in your QuickPitik inbox until $appealBy. " +
            "A person at QuickPitik decides every appeal. If you win, we check every photo and put the good ones back on sale. " +
            "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals).",
    )

    // V2-T13.
    fun wrongEvent(eventName: String, batchPhotos: Int, eventDate: String, appealBy: String) = RelevanceNote(
        "Your photos in $eventName are off sale",
        "Many of your $batchPhotos photos in $eventName seem to be from a different event: their camera dates are far from the " +
            "event's date ($eventDate), or the words printed in them name a different race. " +
            "So our system has taken all your photos in this event off sale. Photos you add to this event are taken off sale too. " +
            "If your camera's clock was set wrong, or the photos do belong to this event, you can appeal from this message in " +
            "your QuickPitik inbox until $appealBy. A person at QuickPitik decides every appeal. " +
            "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals).",
    )

    // V2-T14. Names no other photographer and does not say "copied from".
    fun copied(eventName: String, batchPhotos: Int, appealBy: String) = RelevanceNote(
        "Photos off sale and payouts on hold",
        "Many of your $batchPhotos photos in $eventName match photos that another photographer uploaded. " +
            "So our system has taken all your photos in this event off sale, and your payouts are on hold. " +
            "The hold covers all your payouts, not only this event's sales. Your earnings stay in your balance. " +
            "An admin has been told and will review it. The hold ends when the review is finished or you win an appeal. " +
            "If you took these photos yourself, you can appeal from this message in your QuickPitik inbox until $appealBy. " +
            "A person at QuickPitik decides every appeal. " +
            "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals).",
    )

    // V2-T15.
    fun blurry(eventName: String, scannedCount: Int, blurryCount: Int) = RelevanceNote(
        if (blurryCount == 1) "1 photo looks blurry" else "$blurryCount photos look blurry",
        "We checked $scannedCount of your photos in $eventName. " +
            (if (blurryCount == 1) "One of them looks blurry." else "$blurryCount of them look blurry.") +
            " They are still on sale. Review them, and hide any you do not want to sell. You can show them again later.",
    )

    // V2-T16, after a full check finds fewer problems (scan flip, Mixed to Fits).
    fun restoredAfterScan(eventName: String) = RelevanceNote(
        "Photos back on sale",
        "We checked every one of your photos in $eventName, and fewer look blurry, very dark or not of a race than our first " +
            "check showed. So the photos we took off sale are going back on sale.",
    )

    // V2-T16, after a new check lifts a whole-event hide.
    fun restoredAfterRecheck(eventName: String) = RelevanceNote(
        "Photos back on sale",
        "We checked your photos in $eventName again after your new uploads. The photos we took off sale are going back on " +
            "sale, except those that still look blurry, very dark or not of a race.",
    )

    // V2-T17 ("Session 2" choice 6).
    fun appealReceived(eventName: String, copied: Boolean) = RelevanceNote(
        "Appeal received",
        "We received your appeal about your photos in $eventName. A person at QuickPitik will look at photos from the check " +
            "and decide within 14 days. Until then, " +
            if (copied) "the photos stay off sale and your payouts stay on hold." else "the photos stay off sale.",
    )

    // V2-T18.
    fun appealWon(eventName: String, wholeBatch: Boolean) = RelevanceNote(
        "Appeal accepted",
        "An admin looked at your appeal about $eventName and agreed with you. " + if (wholeBatch) {
            "We are now checking every photo and putting the good ones back on sale. Photos that look blurry, very dark or " +
                "not of a race stay off sale. This does not count against you. If your payouts were on hold because of this, " +
                "the hold has ended."
        } else {
            "The photos we took off sale are going back on sale. This does not count against you."
        },
    )

    // V2-T19 (no admin's reason, "Session 2" choice 7). "For a Copied case:" marks the payout sentence.
    fun appealLost(eventName: String, copied: Boolean) = RelevanceNote(
        "Appeal not accepted",
        "An admin looked at your appeal about $eventName and decided the photos stay off sale. " +
            "When all your photos in the event were off sale, this counts as a confirmed case (see the Terms, Appeals). " +
            (if (copied) "Your payouts stay on hold until an admin finishes the review. " else "") +
            "This is QuickPitik's final decision on the appeal. You can still complain to the National Privacy Commission " +
            "or the Department of Trade and Industry.",
    )

    // V2-T20. windowMonths is app.relevance.strikes.window-months (6).
    fun secondCase(eventName: String, windowMonths: Int) = RelevanceNote(
        "Second confirmed case",
        "Your photos in $eventName were taken off sale, and that decision is now final. This is the second confirmed case " +
            "on your account in $windowMonths months. If there is a third within $windowMonths months, an admin will decide " +
            "whether to suspend your account. Nothing is suspended automatically. To avoid this, upload only your own photos, " +
            "to the event they were taken at, and leave out blurry or very dark shots.",
    )

    // V2-T23, with its conditional sentence: new uploads join an admin's whole-batch hide (findHoldTargets gives the case
    // held_by ADMIN and action HIDE_ALL; the tick's hideBatch holds every photo of the batch).
    fun adminHide(eventName: String, appealBy: String) = RelevanceNote(
        "Your photos in $eventName are off sale",
        "Our automatic check could not tell whether your photos in $eventName belong to this event, so a person at QuickPitik " +
            "looked at them and has taken all your photos in this event off sale. Photos you add to this event are taken off " +
            "sale too. If you think this is wrong, you can appeal from this message in your QuickPitik inbox until $appealBy. " +
            "A person at QuickPitik decides every appeal. If you win, we check every photo and put the good ones back on sale.",
    )

    // V2-T24, approved by the owner 2026-11-16. An admin's hide of a no-consent Junk dump, Wrong event or Copied
    // batch (RV-4.5a); new uploads join it as they join every admin whole-batch hide.
    fun adminHideAfterCheck(eventName: String, appealBy: String) = RelevanceNote(
        "Your photos in $eventName are off sale",
        "Our automatic check found a problem with your photos in $eventName. A person at QuickPitik looked at the results " +
            "and has taken all your photos in this event off sale. Photos you add to this event are taken off sale too. " +
            "If you think this is wrong, you can appeal from this message in your QuickPitik inbox until $appealBy. " +
            "A person at QuickPitik decides every appeal. If you win, we check every photo and put the good ones back on sale.",
    )
}
