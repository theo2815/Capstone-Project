import { describe, expect, it } from "vitest";
import { alertKindLabel, caseActions, verdictNoteLabel, type RelevanceCaseDetail, type RelevanceCaseRow } from "./admin-relevance";

// RVF-19: the closed-case restore (RVF-11), the no-consent mark (RVF-9) and the words-only labels (RVF-7, W3).
const row: RelevanceCaseRow = {
  caseId: "c1", eventId: "e1", eventName: "Cebu City Marathon 2003", photographerHandle: "juanshots",
  photographerName: "Juan Cruz", verdict: "JUNK_DUMP", verdictNote: null, action: "HIDE_ALL", appliedMode: "AUTO_HIDE",
  wouldHave: false, batchPhotos: 420, junkPercent: 81, decidedAt: "2003-02-14T03:00:00Z",
};
type Detail = Pick<RelevanceCaseDetail, "case" | "stage" | "payoutHoldAt" | "payoutHoldPlaced" | "closeReason">;
const detail = (r: RelevanceCaseRow, stage = "APPLIED", closeReason: string | null = null): Detail =>
  ({ case: r, stage, payoutHoldAt: null, payoutHoldPlaced: false, closeReason });
const recordOnly = (extra: Partial<RelevanceCaseRow>): RelevanceCaseRow =>
  ({ ...row, appliedMode: "RECORD_ONLY", wouldHave: true, ...extra });

describe("caseActions", () => {
  it("a closed case with a releasable hold can be restored", () => {
    // The junk a won appeal left held, or a superseded case: the server takes it while the batch has no current case.
    for (const reason of ["APPEAL_WON", "SUPERSEDED", "RESTORED_BY_ADMIN", null]) {
      expect(caseActions(detail(row, "CLOSED", reason), true)).toEqual(["restore"]);
    }
  });

  it("a closed case with nothing releasable offers no restore", () => {
    expect(caseActions(detail(row, "CLOSED", "APPEAL_WON"), false)).toEqual([]);
  });

  it("a no-consent case with its own note still offers the no-consent actions", () => {
    expect(caseActions(detail(recordOnly({ verdictNote: "SUSPICIOUS_WORDS", noConsent: true })), false)).toEqual(["hide_batch"]);
    expect(caseActions(detail(recordOnly({ verdict: "COPIED", verdictNote: "SUSPICIOUS_WORDS", noConsent: true })), false))
      .toEqual(["hide_batch", "hold_payouts"]);
    // Not marked: the note alone decides nothing.
    expect(caseActions(detail(recordOnly({ verdictNote: "SUSPICIOUS_WORDS", noConsent: false })), false)).toEqual([]);
  });

  it("an older server without noConsent reads the note as before", () => {
    expect(caseActions(detail(recordOnly({ verdictNote: "NO_CONSENT" })), false)).toEqual(["hide_batch"]);
    expect(caseActions(detail(recordOnly({ verdictNote: null })), false)).toEqual([]);
  });
});

it("the words-only note and alert have labels", () => {
  expect(verdictNoteLabel("WORDS_ONLY")).toBe(
    "The checker called this a wrong event from printed words only. The camera dates do not show it, so nothing was hidden. Look at the batch. In auto-hide you can hide it by hand if it is from another event.",
  );
  expect(alertKindLabel("WORDS_ONLY")).toBe("Wrong event on words only");
});
