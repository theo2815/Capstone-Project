import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";

const css = readFileSync(path.join(__dirname, "globals.css"), "utf8");

// WCAG 2 relative luminance and contrast ratio.
function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

it("warning text reaches 4.5:1 on surface, paper and paper-deep", () => {
  const value = css.match(/--warning-text:\s*(#[0-9A-Fa-f]{6})/)?.[1];
  expect(value).toBeDefined();
  for (const bg of ["#FFFFFF", "#F8F5EE", "#EFEAE0"]) {
    expect(contrast(value!, bg), `${value} on ${bg}`).toBeGreaterThanOrEqual(4.5);
  }
});

it("caption text (--slate-soft) reaches 4.5:1 on surface, paper and paper-deep", () => {
  const value = css.match(/--slate-soft:\s*(#[0-9A-Fa-f]{6})/)?.[1];
  expect(value).toBeDefined();
  for (const bg of ["#FFFFFF", "#F8F5EE", "#EFEAE0"]) {
    expect(contrast(value!, bg), `${value} on ${bg}`).toBeGreaterThanOrEqual(4.5);
  }
});

it("error red (--error) reaches 4.5:1 on surface, paper and paper-deep", () => {
  const value = css.match(/--error:\s*(#[0-9A-Fa-f]{6})/)?.[1];
  expect(value).toBeDefined();
  for (const bg of ["#FFFFFF", "#F8F5EE", "#EFEAE0"]) {
    expect(contrast(value!, bg), `${value} on ${bg}`).toBeGreaterThanOrEqual(4.5);
  }
});

it("the admin's shared parts use the warning-text token, not raw amber", () => {
  const files = ["admin-status-pill.tsx", "admin-form-fields.tsx", "admin-payout-row.tsx", "admin-payout-bulk-bar.tsx"];
  const raw = files.filter((f) => /amber-\d/.test(readFileSync(path.join(__dirname, "../components/admin", f), "utf8")));
  expect(raw).toEqual([]);
});

// Reads source text only: a class joined another way (["text", tone].join("-")) or a colour in an inline style needs a rendered-page check.
const src = path.join(__dirname, "..");
const sources = readdirSync(src, { recursive: true, encoding: "utf8" })
  .map((f) => f.replace(/\\/g, "/"))
  .filter((f) => /\.[cm]?[jt]sx?$/.test(f) && !/\.test\.[cm]?[jt]sx?$/.test(f));
const read = (f: string) => readFileSync(path.join(src, f), "utf8");

it("warning text uses the warning-text token, not the 3:1 text-warning", () => {
  const raw = sources.filter((f) => /text-warning(?!-text)/.test(read(f)));
  expect(raw).toEqual([]);
});

it("no text uses the border colour line-strong (1.39:1 on paper)", () => {
  const raw = sources.filter((f) => /text-line-strong/.test(read(f)));
  expect(raw).toEqual([]);
});

it("no text- class is built at runtime, where the warning test above cannot see it", () => {
  const built = sources.filter((f) => /\btext-(\$\{|["'`]\s*(\+|\.concat\())/.test(read(f)));
  expect(built).toEqual([]);
});
