import { expect, it } from "vitest";
import type { PhotographerRelevanceAppeal, PhotographerRelevanceCase } from "@/lib/api-photographer";
import { appealText } from "@/lib/photographer-relevance-copy";

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

it("a live hold on a case that is not Copied still shows the payout sentence, as the inbox notes do", () => {
  for (const verdict of ["JUNK_DUMP", "WRONG_EVENT"] as const) {
    expect(appealText(copiedCase("OPEN", { verdict, payoutHoldBinds: true }))!.body).toContain(ON_HOLD_OPEN);
    expect(appealText(copiedCase("LOST", { verdict, payoutHoldBinds: true }))!.body).toContain(ON_HOLD_LOST);
    // An older server without the field keeps the verdict rule.
    expect(appealText(copiedCase("OPEN", { verdict }))!.body).not.toContain(ON_HOLD_OPEN);
    expect(appealText(copiedCase("LOST", { verdict }))!.body).not.toContain(ON_HOLD_LOST);
  }
});
