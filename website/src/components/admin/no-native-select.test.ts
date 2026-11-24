import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";

// Admin selects use DropdownSelect (components/ui/dropdown.tsx), so every list and form field looks and works the same (FT-31).
const SRC = path.resolve(__dirname, "../..");
// A select that must stay native goes here, with its reason. FT-32 moved the last two.
const ALLOWED: string[] = [];

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [full] : [];
  });
}

it("no admin file uses a native select outside the allow-list", () => {
  const hits = [path.join(SRC, "app/admin"), path.join(SRC, "components/admin")]
    .flatMap(sourceFiles)
    .map((file) => path.relative(SRC, file).replaceAll("\\", "/"))
    .filter((rel) => !ALLOWED.includes(rel) && readFileSync(path.join(SRC, rel), "utf8").includes("<select"));
  expect(hits).toEqual([]);
});

it("every allow-listed file still has a native select, so a stale entry cannot hide a new one", () => {
  expect(ALLOWED.filter((rel) => !readFileSync(path.join(SRC, rel), "utf8").includes("<select"))).toEqual([]);
});
