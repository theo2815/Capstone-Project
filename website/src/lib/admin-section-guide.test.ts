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

it("the Runners tip stays under 72 characters (the palette ranking depends on it)", () => {
  expect(getAdminSectionGuideByHref(ROUTES.ADMIN_RUNNERS)!.body.length).toBeLessThan(72);
});

it("the Inbox line uses the one phrase for waiting", () => {
  expect(getAdminSectionGuideByHref(ROUTES.ADMIN_INBOX)!.body).toBe("Everything waiting for an admin, one queue at a time.");
});
