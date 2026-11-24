import { afterEach, expect, it, vi } from "vitest";
import { ADMIN_ROUTES } from "@/components/admin/admin-rail";
import { ADMIN_QUEUES } from "@/lib/admin-queues";
import { ROUTES } from "@/lib/constants";
import { ADMIN_QUEUE_LINES, ADMIN_SECTION_GUIDE, getAdminSectionGuideByHref } from "./admin-section-guide";

afterEach(() => {
  vi.doUnmock("@/lib/constants");
  vi.resetModules();
});

it("every rail page has a guide entry with an intro", () => {
  const missing = ADMIN_ROUTES.filter((r) => !getAdminSectionGuideByHref(r.href)?.intro?.trim()).map((r) => r.href);
  expect(missing).toEqual([]);
});

it("every Inbox queue has its sentence", () => {
  const missing = ADMIN_QUEUES.filter((q) => !ADMIN_QUEUE_LINES[q.key]?.trim()).map((q) => q.key);
  expect(missing).toEqual([]);
});

it("every 'use … instead' link points at a page in the guide", () => {
  const hrefs = new Set(ADMIN_SECTION_GUIDE.map((e) => e.href));
  const dangling = ADMIN_SECTION_GUIDE.flatMap((e) => (e.instead ?? []).filter((i) => !hrefs.has(i.href)));
  expect(dangling).toEqual([]);
});

it("no sibling points at a gated-off page: with Flags off, nothing links to Flags", async () => {
  vi.doMock("@/lib/constants", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/constants")>()),
    ADMIN_FLAGS_ENABLED: false,
  }));
  const { ADMIN_SECTION_GUIDE: gated } = await import("./admin-section-guide");
  expect(gated).toHaveLength(13);
  expect(gated.flatMap((e) => e.instead ?? []).filter((i) => i.href === ROUTES.ADMIN_FLAGS)).toEqual([]);
});

it("with Flags off, no term on another page is about flags (FU-5)", async () => {
  vi.doMock("@/lib/constants", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/constants")>()),
    ADMIN_FLAGS_ENABLED: false,
  }));
  const { ADMIN_SECTION_GUIDE: gated } = await import("./admin-section-guide");
  const terms = gated.flatMap((e) => e.terms ?? []);
  expect(terms.filter((t) => /flag/i.test(`${t.term} ${t.meaning}`))).toEqual([]);
});

// Rows U5.1 to U5.8 of the admin follow-up plan's wording sheet, typed exactly.
const meaning = (href: string, term: string) =>
  getAdminSectionGuideByHref(href)!.terms?.find((t) => t.term === term)?.meaning;

it("each page defines the words it shows (FU-5)", () => {
  expect(meaning(ROUTES.ADMIN_EVENTS, "Live now")).toBe("race day only. From the next day the event is under Recent.");
  expect(meaning(ROUTES.ADMIN_EVENTS, "Recent")).toBe(
    "every day after race day. An event stays here until its status is Archived.",
  );
  expect(meaning(ROUTES.ADMIN_EVENTS, "Archive")).toBe(
    "events whose status is Archived. Age alone never moves an event here.",
  );
  expect(getAdminSectionGuideByHref(ROUTES.ADMIN_EVENTS)!.intro).toBe(
    "Every event on QuickPitik. Add one, edit its details and price, or hide it. A published event moves from Upcoming to Live now to Recent by itself as its date passes. It moves to Archive only when its status is Archived.",
  );
  expect(meaning(ROUTES.ADMIN_FLAGS, "Escalate")).toBe(
    "marks a flag for a second look. Nobody is told, and any admin can still act on it.",
  );
  expect(meaning(ROUTES.ADMIN_FLAGS, "Escalated")).toBe("flags marked for a second look.");
  expect(meaning(ROUTES.ADMIN_FLAGS, "Hold payouts")).toBe(
    "on an older Photo Relevance flag: stops the photographer's payouts until the flag is closed. The photographer is told.",
  );
  expect(meaning(ROUTES.ADMIN_OVERVIEW, "Overdue")).toBe("past the due date on the dispute.");
  expect(meaning(ROUTES.ADMIN_SALES, "Refund pulse")).toBe(
    "refunds and disputes: how much went back to runners, and what is still open.",
  );
  expect(meaning(ROUTES.ADMIN_SALES, "Escalated")).toBe("open disputes marked for more review.");
  expect(meaning(ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS, "Source")).toBe(
    "who filed the request: the runner, or support on their behalf.",
  );
});

it("the Runners tip stays under 72 characters (the palette ranking depends on it)", () => {
  expect(getAdminSectionGuideByHref(ROUTES.ADMIN_RUNNERS)!.body.length).toBeLessThan(72);
});

it("the Inbox line uses the one phrase for waiting", () => {
  expect(getAdminSectionGuideByHref(ROUTES.ADMIN_INBOX)!.body).toBe("Everything waiting for an admin, one queue at a time.");
});

it("the Disputes line says to decide before a dispute is overdue (FT-18)", () => {
  expect(getAdminSectionGuideByHref(ROUTES.ADMIN_DISPUTES)!.body).toBe("Refund requests and complaints from runners. Decide each one before it is overdue.");
});
