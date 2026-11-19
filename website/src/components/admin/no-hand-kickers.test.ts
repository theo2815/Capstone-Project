import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";

// Kicker type goes through <Kicker> or KICKER_SIZE_CLASS, so one retune reaches every admin label (X8).
const HAND_KICKER = "font-mono uppercase tracking-[0.14em] text-[14px]";
const SRC = path.resolve(__dirname, "../..");

function tsxFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return tsxFiles(full);
    return name.endsWith(".tsx") ? [full] : [];
  });
}

it("no admin file hand-copies the kicker class", () => {
  const hits = [path.join(SRC, "app/admin"), path.join(SRC, "components/admin")]
    .flatMap(tsxFiles)
    .flatMap((file) =>
      readFileSync(file, "utf8")
        .split("\n")
        .flatMap((line, i) => (line.includes(HAND_KICKER) ? [`${path.relative(SRC, file)}:${i + 1}`] : [])),
    );
  expect(hits).toEqual([]);
});
