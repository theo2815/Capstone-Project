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

it("the admin's shared parts use the warning-text token, not raw amber", () => {
  const files = ["admin-status-pill.tsx", "admin-form-fields.tsx", "admin-payout-row.tsx", "admin-payout-bulk-bar.tsx"];
  const raw = files.filter((f) => /amber-\d/.test(readFileSync(path.join(__dirname, "../components/admin", f), "utf8")));
  expect(raw).toEqual([]);
});

it("warning text uses the warning-text token, not the 3:1 text-warning", () => {
  const src = path.join(__dirname, "..");
  // shortcut: billing and settings are exempt while PR #231 holds them; AX-52 removes this exemption once #231 merges or closes
  const exempt = ["app/dashboard/billing/page.tsx", "app/dashboard/settings/page.tsx"];
  const raw = readdirSync(src, { recursive: true, encoding: "utf8" })
    .map((f) => f.replace(/\\/g, "/"))
    .filter((f) => /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f) && !exempt.includes(f))
    .filter((f) => /text-warning(?!-text)/.test(readFileSync(path.join(src, f), "utf8")));
  expect(raw).toEqual([]);
});
