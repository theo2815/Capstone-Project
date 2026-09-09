import { describe, expect, it } from "vitest";
import { decodeBase64Url, isTokenExpired } from "./auth";

function createJwt(payload: Record<string, unknown>): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  const body = btoa(JSON.stringify(payload))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  return `${header}.${body}.signature`;
}

describe("decodeBase64Url", () => {
  it("decodes regular base64 strings", () => {
    const raw = JSON.stringify({ hello: "world" });
    const b64 = btoa(raw);
    expect(decodeBase64Url(b64)).toBe(raw);
  });

  it("decodes base64url strings with - and _ and omitted padding", () => {
    // String whose base64 encoding contains characters mapped in Base64URL
    const payload = JSON.stringify({ sub: "user-123_abc", exp: 1234567890 });
    const b64url = btoa(payload)
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
    expect(decodeBase64Url(b64url)).toBe(payload);
  });

  it("throws on invalid base64url length with pad % 4 === 1", () => {
    expect(() => decodeBase64Url("a")).toThrow();
  });
});

describe("isTokenExpired", () => {
  it("returns false for a future expiration timestamp even with Base64URL characters", () => {
    const futureExp = Math.floor((Date.now() + 60 * 60 * 1000) / 1000);
    const token = createJwt({ sub: "user-test_runner-01", exp: futureExp });
    expect(isTokenExpired(token)).toBe(false);
  });

  it("returns true for a past expiration timestamp", () => {
    const pastExp = Math.floor((Date.now() - 60 * 60 * 1000) / 1000);
    const token = createJwt({ sub: "user-test_runner-01", exp: pastExp });
    expect(isTokenExpired(token)).toBe(true);
  });

  it("returns true for malformed tokens or invalid JSON", () => {
    expect(isTokenExpired("")).toBe(true);
    expect(isTokenExpired("not-a-token")).toBe(true);
    expect(isTokenExpired("header.invalid-payload.sig")).toBe(true);
  });

  it("returns true for token missing exp field", () => {
    const token = createJwt({ sub: "user-no-exp" });
    expect(isTokenExpired(token)).toBe(true);
  });
});
