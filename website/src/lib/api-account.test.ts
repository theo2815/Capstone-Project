import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { mintDesktopAuthCode } from "./api-account";

afterEach(() => vi.restoreAllMocks());

describe("mintDesktopAuthCode", () => {
  it("posts the challenge to the desktop codes route and returns the code", async () => {
    const post = vi.spyOn(api, "post").mockResolvedValue({ code: "c".repeat(43), expiresInSeconds: 120 });

    const result = await mintDesktopAuthCode("x".repeat(43));

    expect(post).toHaveBeenCalledWith("/auth/desktop/codes", { codeChallenge: "x".repeat(43) });
    expect(result).toEqual({ code: "c".repeat(43), expiresInSeconds: 120 });
  });
});
