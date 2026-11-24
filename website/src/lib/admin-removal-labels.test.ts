import { expect, it } from "vitest";
import { removalActionLabel, removalStepLine } from "./admin-removal-labels";
import { formatSalesDate } from "./format";
import { draftCase, openCase, ownerReview, removalItem, runnerCase } from "@/components/admin/photo-removal/removal-fixtures";

const NOW = Date.parse("2026-11-24T00:00:00Z");
const SOON = "2026-11-26T03:00:00Z";
const LATER = "2026-11-27T03:00:00Z";
const PAST = "2026-11-20T03:00:00Z";

// FT-32 (C-74, W32.1): one line under the header names the step and who it waits for.
it("a draft is step 1", () => {
  expect(removalStepLine(draftCase, NOW)).toBe("Step 1 of 4. Choose the photos this request is about, then verify them.");
});

it("an open request waiting on owners names how many and until when", () => {
  const request = {
    ...openCase,
    items: [removalItem("a1", "A"), removalItem("b1", "B"), removalItem("c1", "C", { ownerResponse: "AGREE" })],
    reviews: [ownerReview("A", { deadline: SOON }), ownerReview("B", { deadline: LATER }), ownerReview("C", { deadline: LATER })],
  };
  expect(removalStepLine(request, NOW)).toBe(
    `Step 2 of 4. Waiting for 2 photographers to answer, until ${formatSalesDate(LATER)}. You can decide earlier with an urgent reason.`);
  const one = { ...request, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline: SOON })] };
  expect(removalStepLine(one, NOW)).toBe(
    `Step 2 of 4. Waiting for 1 photographer to answer, until ${formatSalesDate(SOON)}. You can decide earlier with an urgent reason.`);
});

it("an open request whose owners answered, ran out of time or cannot be reached is step 3", () => {
  const ready = "Step 3 of 4. Ready for your decision on each photo.";
  const answered = { ...openCase, items: [removalItem("a1", "A", { ownerResponse: "OBJECT" })], reviews: [ownerReview("A", { deadline: LATER })] };
  const overdue = { ...openCase, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline: PAST, overdue: true })] };
  const lapsed = { ...openCase, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline: PAST })] };
  const unreachable = { ...openCase, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline: LATER, unavailableReason: "Account closed" })] };
  const noOwner = { ...openCase, items: [removalItem("a1", null)], reviews: [] };
  for (const request of [answered, overdue, lapsed, unreachable, noOwner]) expect(removalStepLine(request, NOW)).toBe(ready);
});

// FU-12 (U12.3): an owner with no deadline yet has not been told; the line names who is waited for, with no date.
it("an open request whose owner has no deadline yet waits with no date", () => {
  const untold = { ...openCase, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline: null, notifiedAt: null })] };
  expect(removalStepLine(untold, NOW)).toBe("Step 2 of 4. Waiting for 1 photographer to answer.");
  const mixed = {
    ...openCase, items: [removalItem("a1", "A"), removalItem("b1", "B")],
    reviews: [ownerReview("A", { deadline: null, notifiedAt: null }), ownerReview("B", { deadline: SOON })],
  };
  expect(removalStepLine(mixed, NOW)).toBe("Step 2 of 4. Waiting for 2 photographers to answer.");
});

it("a closing request says the files are being deleted only while a removal is authorized", () => {
  const removing = { ...openCase, status: "RESOLVING" as const, items: [removalItem("a1", "A", { state: "REMOVAL_AUTHORIZED" })] };
  expect(removalStepLine(removing, NOW)).toBe("Step 4 of 4. Removal authorized. The files are being deleted.");
  // FU-12 (U12.2): a cancel also passes through Closing, with nothing being deleted.
  const cancelling = { ...openCase, status: "RESOLVING" as const, items: [removalItem("a1", "A", { state: "EXCLUDED" })] };
  expect(removalStepLine(cancelling, NOW)).toBe("Step 4 of 4. Closing the request. No photo is being removed.");
});

it("a resolved request is finished", () => {
  expect(removalStepLine({ ...openCase, status: "RESOLVED" }, NOW)).toBe("Finished. Every photo was decided.");
});

// FU-12 (U12.1): only an admin cancels a support request, so the line is true there.
it("a cancelled support request says what went back on sale and who was told", () => {
  expect(removalStepLine({ ...openCase, status: "CANCELLED" }, NOW)).toBe("Cancelled. Photos hidden for this request went back on sale, " +
    "unless another request still holds them. A runner who filed it was told the photos were kept. Photographers were not told.");
});

// A runner's request also ends Cancelled when the runner withdraws it or no photo matched; they hear other words then,
// and the admin data does not say which, so no line shows.
it("a cancelled runner request shows no line", () => {
  expect(removalStepLine({ ...runnerCase, status: "CANCELLED" }, NOW)).toBeNull();
});

// FU-12 (U12.8): history labels say "request" and "possible match".
it("history labels say request and possible match", () => {
  expect(["CREATED", "RESOLVED", "CANCELLED", "SUGGESTED", "SUGGESTION_DISMISSED"].map(removalActionLabel)).toEqual(
    ["Request created", "Request finished", "Request cancelled", "Possible match", "Possible match removed"]);
});
