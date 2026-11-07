import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";

// The root layout's title template already appends "| QuickPitik", so a page
// title that ends with it shows the brand twice in the tab.
function pages(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return pages(full);
    return /^(page|layout)\.tsx?$/.test(name) && full !== path.join(__dirname, "layout.tsx") ? [full] : [];
  });
}

it("no page title repeats the brand that the root template adds", () => {
  const doubled = pages(__dirname).filter((file) =>
    /title:\s*["'`][^"'`]*\|\s*QuickPitik["'`]/.test(readFileSync(file, "utf8")),
  );
  expect(doubled.map((f) => path.relative(__dirname, f))).toEqual([]);
});
