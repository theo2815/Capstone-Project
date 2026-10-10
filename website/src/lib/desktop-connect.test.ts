import { describe, expect, it } from "vitest";
import { buildDesktopCallbackUrl, isDesktopConnectParam } from "./desktop-connect";

const PARAM = "A-z0_9".repeat(7) + "Q";

describe("isDesktopConnectParam", () => {
  it("accepts exactly 43 base64url characters", () => {
    expect(PARAM).toHaveLength(43);
    expect(isDesktopConnectParam(PARAM)).toBe(true);
  });

  it("rejects other lengths, other alphabets and missing values", () => {
    expect(isDesktopConnectParam(PARAM.slice(1))).toBe(false);
    expect(isDesktopConnectParam(PARAM + "a")).toBe(false);
    expect(isDesktopConnectParam("a".repeat(42) + "=")).toBe(false);
    expect(isDesktopConnectParam("a".repeat(42) + "/")).toBe(false);
    expect(isDesktopConnectParam("")).toBe(false);
    expect(isDesktopConnectParam(null)).toBe(false);
    expect(isDesktopConnectParam(undefined)).toBe(false);
  });
});

describe("buildDesktopCallbackUrl", () => {
  it("builds the batchmyphotos link with both values encoded", () => {
    expect(buildDesktopCallbackUrl("c".repeat(43), "s".repeat(43))).toBe(
      `batchmyphotos://quickpitik/callback?code=${"c".repeat(43)}&state=${"s".repeat(43)}`,
    );
    expect(buildDesktopCallbackUrl("a b", "x&y")).toBe("batchmyphotos://quickpitik/callback?code=a%20b&state=x%26y");
  });
});
