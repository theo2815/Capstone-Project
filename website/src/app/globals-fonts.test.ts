import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";

const css = readFileSync(path.join(__dirname, "globals.css"), "utf8");
const layout = readFileSync(path.join(__dirname, "layout.tsx"), "utf8");

// next/font sets these variables on <body> (layout.tsx). The `@theme inline`
// aliases (--font-sans, --font-hero, --font-mono, --font-display) are inlined
// into Tailwind utilities and never exist at run time, so a hand-written rule
// that reads them silently falls back to the system font.
const loadedFontVars = new Set([...layout.matchAll(/variable:\s*"(--font-[\w-]+)"/g)].map((m) => m[1]));

it.each(["body", ".font-hero", ".kicker"])("%s reads a font variable that next/font really sets", (selector) => {
  const escaped = selector.replace(".", "\\.");
  const rule = css.match(new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]*)\\}`))?.[1];
  expect(rule, `${selector} rule`).toBeDefined();
  const family = rule!.match(/font-family:\s*([^;]+);/)?.[1] ?? "";
  const used = [...family.matchAll(/var\((--[\w-]+)\)/g)].map((m) => m[1]);
  expect(used.length).toBeGreaterThan(0);
  for (const v of used) expect(loadedFontVars, `${selector} uses ${v}`).toContain(v);
});
