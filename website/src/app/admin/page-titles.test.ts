import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";
import { ADMIN_ROUTES } from "@/components/admin/admin-rail";

// D2: every admin page's H1 is its rail label with a full stop.
it("every admin page heading starts with its rail label", () => {
  const mismatches = ADMIN_ROUTES.flatMap(({ href, label }) => {
    const source = readFileSync(path.join(__dirname, href.replace(/^\/admin\//, ""), "page.tsx"), "utf8");
    const heading = source.match(/<h1[^>]*>([\s\S]*?)<\/h1>/)?.[1].trim();
    return heading === `${label}.` ? [] : [`${href}: "${heading}" should be "${label}."`];
  });
  expect(mismatches).toEqual([]);
});
