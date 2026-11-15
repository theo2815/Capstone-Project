import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { acceptTerms, mintDesktopAuthCode } from "./api-account";

afterEach(() => vi.restoreAllMocks());

describe("mintDesktopAuthCode", () => {
  it("posts the challenge to the desktop codes route and returns the code", async () => {
    const post = vi.spyOn(api, "post").mockResolvedValue({ code: "c".repeat(43), expiresInSeconds: 120 });

    const result = await mintDesktopAuthCode("x".repeat(43));

    expect(post).toHaveBeenCalledWith("/auth/desktop/codes", { codeChallenge: "x".repeat(43) });
    expect(result).toEqual({ code: "c".repeat(43), expiresInSeconds: 120 });
  });
});

describe("acceptTerms", () => {
  it("posts the version to the terms-acceptance route and returns the user", async () => {
    const user = { id: "u-1", termsAcceptanceRequired: false };
    const post = vi.spyOn(api, "post").mockResolvedValue(user);

    expect(await acceptTerms("1.6")).toBe(user);
    expect(post).toHaveBeenCalledWith("/me/terms-acceptance", { version: "1.6" });
  });
});
