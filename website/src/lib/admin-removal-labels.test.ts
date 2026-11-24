import { expect, it } from "vitest";
import { removalStepLine } from "./admin-removal-labels";
import { formatSalesDate } from "./format";
import { draftCase, openCase, ownerReview, removalItem } from "@/components/admin/photo-removal/removal-fixtures";

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

it("an open request whose owner has no deadline yet shows no line", () => {
  // Not told yet: neither "waiting until" (no date) nor "ready" (the decision still needs an urgent reason).
  const untold = { ...openCase, items: [removalItem("a1", "A")], reviews: [ownerReview("A", { deadline: null, notifiedAt: null })] };
  expect(removalStepLine(untold, NOW)).toBeNull();
});

it("a closing request says the files are being deleted only while a removal is authorized", () => {
  const removing = { ...openCase, status: "RESOLVING" as const, items: [removalItem("a1", "A", { state: "REMOVAL_AUTHORIZED" })] };
  expect(removalStepLine(removing, NOW)).toBe("Step 4 of 4. Removal authorized. The files are being deleted.");
  // A cancel also passes through Closing: then nothing is being deleted, so no line.
  const cancelling = { ...openCase, status: "RESOLVING" as const, items: [removalItem("a1", "A", { state: "EXCLUDED" })] };
  expect(removalStepLine(cancelling, NOW)).toBeNull();
});

it("a resolved request is finished", () => {
  expect(removalStepLine({ ...openCase, status: "RESOLVED" }, NOW)).toBe("Finished. Every photo was decided.");
});

// Held: a request cancelled while open had its photos hidden, so "Nothing was hidden" is not always true.
it("a cancelled request shows no line", () => {
  expect(removalStepLine({ ...openCase, status: "CANCELLED" }, NOW)).toBeNull();
});
