import type { PhotographerRelevanceCase } from "@/lib/api-photographer";

// RV-3.4a: every sentence the photographer's case page shows, copied character for character from the approved RV-L2
// texts (vault journal 2026-11-15-rv-l2-relevance-v2-wording, "Final texts" and "Session 2"). Code fills in values
// only. The note's "you can appeal from this message in your QuickPitik inbox until …" sentence is left out: on this
// page the appeal form's own line (V2-T22) says it. Never an input: the model's reason (G12).

export interface CaseText {
  title: string;
  /** What happened (the note's sentences before its appeal line). */
  body: string[];
  /** What an appeal leads to; shown only while the form is open. */
  outcome: string[];
}

const CONFIRMED = "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals).";
const WIN_RESCAN = "If you win, we check every photo and put the good ones back on sale.";
const ALL_OFF = "So our system has taken all your photos in this event off sale.";
const NEW_TOO = "Photos you add to this event are taken off sale too.";
const checked = (c: PhotographerRelevanceCase) =>
  `We checked ${c.sampleSize ?? 0} of your ${c.batchPhotos} photos in ${c.eventName}, and ${c.junkCount ?? 0} of them look blurry, very dark or not of a race.`;
const offSaleIn = (eventName: string) => `Your photos in ${eventName} are off sale`;

// The note texts of a hiding case, by verdict (V2-T11 to V2-T14, V2-T23). `eventDate` is V2-T13's {eventDate}; without
// it the wrong-event body is left out whole, so its "So our system…" sentence never stands first.
export function hiddenCaseText(c: PhotographerRelevanceCase, eventDate: string | null): CaseText {
  switch (c.verdict) {
    case "MIXED":
      return { title: "Some of your photos are off sale", outcome: [], body: [checked(c),
        "Our system is now checking every photo and taking the ones that look like that off sale.", "Your other photos stay on sale."] };
    case "WRONG_EVENT":
      return { title: offSaleIn(c.eventName), outcome: [CONFIRMED], body: eventDate ? [
        `Many of your ${c.batchPhotos} photos in ${c.eventName} seem to be from a different event: their camera dates are far from the event's date (${eventDate}), or the words printed in them name a different race.`,
        ALL_OFF, NEW_TOO] : [] };
    case "COPIED":
      return { title: "Photos off sale and payouts on hold", outcome: [CONFIRMED], body: [
        `Many of your ${c.batchPhotos} photos in ${c.eventName} match photos that another photographer uploaded.`,
        "So our system has taken all your photos in this event off sale, and your payouts are on hold.",
        "The hold covers all your payouts, not only this event's sales.", "Your earnings stay in your balance.",
        "An admin has been told and will review it.", "The hold ends when the review is finished or you win an appeal."] };
    // A hiding Can't tell case is only ever an admin's hide (RV-2.5b), V2-T23 with its "too" sentence (as RV-3.1 built).
    case "CANT_TELL":
      return { title: offSaleIn(c.eventName), outcome: [WIN_RESCAN], body: [
        `Our automatic check could not tell whether your photos in ${c.eventName} belong to this event, so a person at QuickPitik looked at them and has taken all your photos in this event off sale.`,
        NEW_TOO] };
    case "JUNK_DUMP":
      return { title: offSaleIn(c.eventName), outcome: [WIN_RESCAN, CONFIRMED], body: [checked(c), ALL_OFF, NEW_TOO] };
    // A Fits re-check that lifted a junk dump (G7): the lifted case's V2-T16 note describes this case's state.
    case "FITS":
      return c.action === "HIDE_JUNK" ? shownAgainText(c.eventName, "recheck")
        : { title: offSaleIn(c.eventName), outcome: [], body: [] };
    // No note exists for any other hiding combination: the title alone, no rule sentence.
    default:
      return { title: c.action === "HIDE_JUNK" ? "Some of your photos are off sale" : offSaleIn(c.eventName), outcome: [], body: [] };
  }
}

// V2-T16, with the body for what happened, as RelevanceNotifier picks it: "scan" for a current Fits case with an appeal
// clock (it was Mixed and its full check flipped it, A.4), "recheck" for a case a new check lifted (closed
// RECHECK_LIFTED, or the Fits case that lifted it). Otherwise the title alone.
export function shownAgainText(eventName: string, after: "scan" | "recheck" | null): CaseText {
  return { title: "Photos back on sale", outcome: [], body: after === "scan" ? [
    `We checked every one of your photos in ${eventName}, and fewer look blurry, very dark or not of a race than our first check showed.`,
    "So the photos we took off sale are going back on sale."]
    : after === "recheck" ? [`We checked your photos in ${eventName} again after your new uploads.`,
      "The photos we took off sale are going back on sale, except those that still look blurry, very dark or not of a race."]
    : [] };
}

// V2-T15 in full (RV-3.4b: the page now offers the review and the undo the last two sentences promise).
export function blurryText(c: PhotographerRelevanceCase): CaseText {
  const n = c.scanBlurry ?? 0;
  return { title: n === 1 ? "1 photo looks blurry" : `${n} photos look blurry`, outcome: [], body: [
    `We checked ${c.scanMeasured ?? 0} of your photos in ${c.eventName}.`, n === 1 ? "One of them looks blurry." : `${n} of them look blurry.`,
    "They are still on sale.", "Review them, and hide any you do not want to sell.", "You can show them again later."] };
}

// V2-T17, V2-T18, V2-T19: the appeal's state, as the inbox notes word it ("Session 2" choices 6 and 7).
// RV-4.5a review: as the inbox notes, the payout sentence only while a hold binds, the confirmed-case one only if it counts.
export function appealText(c: PhotographerRelevanceCase): CaseText | null {
  const copied = c.verdict === "COPIED" && c.payoutHoldBinds !== false;
  switch (c.appeal?.status) {
    case "OPEN":
      return { title: "Appeal received", outcome: [], body: [
        `We received your appeal about your photos in ${c.eventName}.`,
        "A person at QuickPitik will look at photos from the check and decide within 14 days.",
        copied ? "Until then, the photos stay off sale and your payouts stay on hold." : "Until then, the photos stay off sale."] };
    case "WON":
      return { title: "Appeal accepted", outcome: [], body: [
        `An admin looked at your appeal about ${c.eventName} and agreed with you.`,
        ...(c.verdict === "MIXED" ? ["The photos we took off sale are going back on sale.", "This does not count against you."] : [
          "We are now checking every photo and putting the good ones back on sale.",
          "Photos that look blurry, very dark or not of a race stay off sale.", "This does not count against you.",
          "If your payouts were on hold because of this, the hold has ended."])] };
    case "LOST":
      return { title: "Appeal not accepted", outcome: [], body: [
        `An admin looked at your appeal about ${c.eventName} and decided the photos stay off sale.`,
        ...(c.lostAppealCounts !== false ? ["When all your photos in the event were off sale, this counts as a confirmed case (see the Terms, Appeals)."] : []),
        ...(copied ? ["Your payouts stay on hold until an admin finishes the review."] : []),
        "This is QuickPitik's final decision on the appeal.",
        "You can still complain to the National Privacy Commission or the Department of Trade and Industry."] };
    default:
      return null;
  }
}

// V2-T22, the appeal form.
export const APPEAL_FORM = {
  title: "Appeal this decision",
  line: (appealBy: string) => `You can appeal until ${appealBy}, 14 days from our message. You can appeal this decision once. A person at QuickPitik will look at photos from the check and decide within 14 days.`,
  label: "Why do you think this decision is wrong?",
  hint: "Up to 1,000 characters. You cannot add files.",
  counter: (typed: number) => `${typed.toLocaleString("en-US")}/1,000`,
  send: "Send appeal",
  cancel: "Cancel",
  ended: "The time to appeal this decision has ended.",
} as const;
export const APPEAL_MAX = 1000;

// No approved sentence exists for these page states (listed as a text gap in RV-3.4a): plain neutral labels only.
export const CASE_LABELS = {
  kicker: "Uploads", // V2-T21's kicker
  loading: "Loading…",
  notFound: "Case not found",
  loadFailed: "Couldn't load",
  retry: "Try again",
  closed: "Closed",
  sending: "Sending…",
} as const;

// The Philippine calendar day of `now`, as YYYY-MM-DD, to compare with appealDeadline (filing is open through it).
export function philippineToday(now = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Manila", year: "numeric", month: "2-digit", day: "2-digit" }).format(now);
}

// RV-3.4b: the blurry-photos review. "Review and hide" is V2-T15's button label; the others have no approved text
// (listed as a gap in RV-3.4b): plain neutral labels.
export const BLURRY_LABELS = {
  review: "Review and hide",
  looksBlurry: "Looks blurry",
  hide: "Hide",
  hiding: "Hiding…",
  hidden: "Hidden",
  undo: "Undo",
  selected: (n: number) => `${n.toLocaleString("en-US")} selected`,
  // One name per tile; aria-pressed says whether it is selected.
  tile: (i: number) => `Photo ${i}`,
  undoPhoto: (i: number) => `Undo photo ${i}`,
} as const;
