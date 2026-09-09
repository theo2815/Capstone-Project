import { describe, expect, it } from "vitest";
import {
  isReservedHandle,
  isValidHandleFormat,
  validateHandle,
} from "./reserved-handles";

describe("reserved-handles", () => {
  it("rejects reserved system and authentication routes", () => {
    expect(isReservedHandle("verify")).toBe(true);
    expect(isReservedHandle("verify-email")).toBe(true);
    expect(isReservedHandle("confirm-email-change")).toBe(true);
    expect(isReservedHandle("reset-password")).toBe(true);
    expect(isReservedHandle("admin")).toBe(true);
    expect(isReservedHandle("login")).toBe(true);
    expect(isReservedHandle("dashboard")).toBe(true);
  });

  it("handles case-insensitive and trimmed handle checks", () => {
    expect(isReservedHandle("  Verify-Email  ")).toBe(true);
    expect(isReservedHandle("CONFIRM-EMAIL-CHANGE")).toBe(true);
    expect(isReservedHandle("ADMIN")).toBe(true);
  });

  it("permits valid unreserved handles", () => {
    expect(isReservedHandle("cebu-sports-photos")).toBe(false);
    expect(isReservedHandle("lensmaster-99")).toBe(false);
    expect(isValidHandleFormat("cebu-sports-photos")).toBe(true);
    expect(validateHandle("cebu-sports-photos")).toBeNull();
  });

  it("rejects invalid handle formats and reserved names in validateHandle", () => {
    expect(validateHandle("")).toBe("Handle is required.");
    expect(validateHandle("ab")).toBe("Handle must be at least 3 characters.");
    expect(validateHandle("verify-email")).toBe("That handle is reserved. Try another.");
    expect(validateHandle("invalid_handle!")).toBe(
      "Use lowercase letters, numbers, and dashes. No spaces or symbols.",
    );
  });
});
