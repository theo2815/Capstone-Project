import { describe, expect, it } from "vitest";

// Throwaway: proves the website-tests job goes red on a failing test. Never merged.
describe("rvf-26 red proof", () => {
  it("fails on purpose", () => {
    expect(1).toBe(2);
  });
});
