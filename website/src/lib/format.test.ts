import { expect, it } from "vitest";
import { formatLongDate } from "./format";

it("date-only keeps a bare date on its own day", () => {
  expect(formatLongDate("2026-11-07", true)).toBe("NOV 07 · 2026");
});

it("date-only still reads a full timestamp (admin seller rows, photographer Joined)", () => {
  // Noon UTC is 2026-11-07 in every time zone from UTC-11 to UTC+11.
  expect(formatLongDate("2026-11-07T12:00:00Z", true)).toBe("NOV 07 · 2026");
});
