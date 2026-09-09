import { describe, expect, it } from "vitest";
import { decodeBase64Url } from "./auth";

describe("decodeBase64Url", () => {
  it("decodes padded base64 JSON", () => {
    const raw = JSON.stringify({ hello: "world" });
    expect(decodeBase64Url(btoa(raw))).toBe(raw);
  });

  it("decodes UTF-8 JSON with both URL-safe characters and omitted padding", () => {
    const payload = JSON.stringify({ name: "José 🏃🏿", sub: "runner" });
    const encoded = Buffer.from(payload, "utf8").toString("base64url");
    expect(encoded).toContain("-");
    expect(encoded).toContain("_");
    expect(encoded).not.toContain("=");
    expect(JSON.parse(decodeBase64Url(encoded))).toEqual(JSON.parse(payload));
  });

  it("rejects malformed base64 and UTF-8", () => {
    for (const encoded of ["a", "%%%", "_w"]) {
      expect(() => decodeBase64Url(encoded)).toThrow();
    }
  });
});
