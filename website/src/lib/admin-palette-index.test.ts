import { existsSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";
import { ADMIN_ROUTES } from "@/components/admin/admin-rail";
import { buildPaletteIndex } from "./admin-palette-index";
import { adminUser } from "@/test/fixtures/admin";

const pages = buildPaletteIndex({ userPool: [], disputes: [], flags: [], payouts: [] }).filter((e) => e.kind === "page");

it("has a seed for every admin page in the rail", () => {
  const routes = new Set(pages.map((p) => p.route));
  for (const r of ADMIN_ROUTES) expect(routes, r.label).toContain(r.href);
});

it("points no seed at a page that does not exist", () => {
  for (const p of pages) {
    const path = new URL(p.route, "http://x").pathname;
    expect(existsSync(join(__dirname, "../app", path, "page.tsx")), p.route).toBe(true);
  }
});

// AX-57: drawer and detail values read as typed, wrap at 375, and link a no-handle photographer by id.
it("a photographer with no handle opens their page by id", () => {
  const row = { ...adminUser, userId: "photographer-9", handle: null };
  const [entry] = buildPaletteIndex({ userPool: [row], disputes: [], flags: [], payouts: [] }).filter((e) => e.kind === "photographer");
  expect(entry.route).toBe("/admin/photographers/photographer-9");
});
