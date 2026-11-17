import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import type { PhotographerRelevanceAppeal, PhotographerRelevanceCase } from "@/lib/api-photographer";
import { appealText, hiddenCaseText } from "@/lib/photographer-relevance-copy";

// RV-4.5b review: the page leaves out the same two sentences as the inbox notes (RelevanceNotices, RV-4.5a).
const copiedCase = (status: string, extra: Partial<PhotographerRelevanceCase> = {}): PhotographerRelevanceCase => ({
  caseId: "case-1", eventId: "event-1", eventName: "Cebu Marathon", verdict: "COPIED", action: "HIDE_ALL", current: true,
  batchPhotos: 400, sampleSize: 300, junkCount: 0, hiddenPhotos: 400, scanMeasured: null, scanBlurry: null,
  noticeAt: "2026-11-10T03:00:00Z", appealDeadline: "2026-11-24", eventDate: "2026-11-08", closeReason: null,
  appeal: { status } as PhotographerRelevanceAppeal, ...extra,
});
const ON_HOLD_OPEN = "Until then, the photos stay off sale and your payouts stay on hold.";
const OFF_SALE_OPEN = "Until then, the photos stay off sale.";
const ON_HOLD_LOST = "Your payouts stay on hold until an admin finishes the review.";
const COUNTS = "When all your photos in the event were off sale, this counts as a confirmed case (see the Terms, Appeals).";

it("an open appeal mentions the payout hold only while one binds the case", () => {
  expect(appealText(copiedCase("OPEN"))!.body).toContain(ON_HOLD_OPEN);
  expect(appealText(copiedCase("OPEN", { payoutHoldBinds: true }))!.body).toContain(ON_HOLD_OPEN);
  const body = appealText(copiedCase("OPEN", { payoutHoldBinds: false }))!.body;
  expect(body).toContain(OFF_SALE_OPEN);
  expect(body).not.toContain(ON_HOLD_OPEN);
});

it("a lost appeal mentions the payout hold only while one binds the case", () => {
  expect(appealText(copiedCase("LOST"))!.body).toContain(ON_HOLD_LOST);
  expect(appealText(copiedCase("LOST", { payoutHoldBinds: true }))!.body).toContain(ON_HOLD_LOST);
  expect(appealText(copiedCase("LOST", { payoutHoldBinds: false }))!.body).not.toContain(ON_HOLD_LOST);
});

it("a lost appeal says it counts as a confirmed case unless the hide was an admin's", () => {
  expect(appealText(copiedCase("LOST"))!.body).toContain(COUNTS);
  expect(appealText(copiedCase("LOST", { lostAppealCounts: true }))!.body).toContain(COUNTS);
  const body = appealText(copiedCase("LOST", { lostAppealCounts: false }))!.body;
  expect(body).not.toContain(COUNTS);
  // Every other sentence stays as it is.
  expect(body).toEqual([
    "An admin looked at your appeal about Cebu Marathon and decided the photos stay off sale.",
    ON_HOLD_LOST,
    "This is QuickPitik's final decision on the appeal.",
    "You can still complain to the National Privacy Commission or the Department of Trade and Industry.",
  ]);
});

// RVF-18 (RVR-5): the "taken off sale" text follows the same two fields.
const hidingCase = (verdict: "JUNK_DUMP" | "WRONG_EVENT" | "COPIED", extra: Partial<PhotographerRelevanceCase> = {}) =>
  ({ ...copiedCase("OPEN"), verdict, junkCount: 240, appeal: null, ...extra });
const ADMIN_BODY = [
  "Our automatic check found a problem with your photos in Cebu Marathon.",
  "A person at QuickPitik looked at the results and has taken all your photos in this event off sale.",
  "Photos you add to this event are taken off sale too.",
];
const WIN_RESCAN = "If you win, we check every photo and put the good ones back on sale.";
const CONFIRMED = "If you do not appeal, or you lose the appeal, this counts as a confirmed case (see the Terms, Appeals).";

it("an admin's hide of a junk dump, wrong event or copied batch shows the V2-T24 sentences and never \"our system\" or \"confirmed case\"", () => {
  for (const verdict of ["JUNK_DUMP", "WRONG_EVENT", "COPIED"] as const) {
    for (const payoutHoldBinds of [true, false]) {
      const text = hiddenCaseText(hidingCase(verdict, { lostAppealCounts: false, payoutHoldBinds }), "NOV 8 · 2026");
      expect(text).toEqual({ title: "Your photos in Cebu Marathon are off sale", body: ADMIN_BODY, outcome: [WIN_RESCAN] });
      const all = [text.title, ...text.body, ...text.outcome].join(" ");
      expect(all).not.toMatch(/our system|confirmed case|payouts/i);
    }
  }
});

it("a copied case with no live hold says nothing about payouts", () => {
  const text = hiddenCaseText(hidingCase("COPIED", { payoutHoldBinds: false }), null);
  expect(text).toEqual({
    title: "Your photos in Cebu Marathon are off sale",
    body: ["Many of your 400 photos in Cebu Marathon match photos that another photographer uploaded.",
      "So our system has taken all your photos in this event off sale."],
    outcome: [CONFIRMED],
  });
});

it("a copied case with a live hold keeps the V2-T14 text", () => {
  const text = hiddenCaseText(hidingCase("COPIED", { payoutHoldBinds: true, lostAppealCounts: true }), null);
  expect(text.title).toBe("Photos off sale and payouts on hold");
  expect(text.body).toContain("So our system has taken all your photos in this event off sale, and your payouts are on hold.");
  expect(text.outcome).toEqual([CONFIRMED]);
});

it("a server that sends neither field reads as before", () => {
  expect(hiddenCaseText(hidingCase("COPIED"), null).title).toBe("Photos off sale and payouts on hold");
  const junk = hiddenCaseText(hidingCase("JUNK_DUMP"), null);
  expect(junk.body).toContain("So our system has taken all your photos in this event off sale.");
  expect(junk.outcome).toEqual([WIN_RESCAN, CONFIRMED]);
  expect(hiddenCaseText(hidingCase("WRONG_EVENT"), "NOV 8 · 2026").outcome).toEqual([CONFIRMED]);
});

it("the V2-T24 strings equal the backend's to the letter", () => {
  const kt = readFileSync(resolve(process.cwd(), "../backend/src/main/kotlin/com/quickpitik/service/relevance/RelevanceNotices.kt"), "utf8");
  const fn = kt.slice(kt.indexOf("fun adminHideAfterCheck"), kt.indexOf("\n    )", kt.indexOf("fun adminHideAfterCheck")));
  const [title, ...bodyParts] = [...fn.matchAll(/"([^"]*)"/g)].map((m) => m[1].replaceAll("$eventName", "Cebu Marathon"));
  const body = bodyParts.join("");
  const text = hiddenCaseText(hidingCase("JUNK_DUMP", { lostAppealCounts: false }), null);
  expect(text.title).toBe(title);
  expect(body.startsWith(text.body.join(" ") + " ")).toBe(true);
  expect(body.endsWith(" " + text.outcome.join(" "))).toBe(true);
});

it("a live hold on a case that is not Copied still shows the payout sentence, as the inbox notes do", () => {
  for (const verdict of ["JUNK_DUMP", "WRONG_EVENT"] as const) {
    expect(appealText(copiedCase("OPEN", { verdict, payoutHoldBinds: true }))!.body).toContain(ON_HOLD_OPEN);
    expect(appealText(copiedCase("LOST", { verdict, payoutHoldBinds: true }))!.body).toContain(ON_HOLD_LOST);
    // An older server without the field keeps the verdict rule.
    expect(appealText(copiedCase("OPEN", { verdict }))!.body).not.toContain(ON_HOLD_OPEN);
    expect(appealText(copiedCase("LOST", { verdict }))!.body).not.toContain(ON_HOLD_LOST);
  }
});
