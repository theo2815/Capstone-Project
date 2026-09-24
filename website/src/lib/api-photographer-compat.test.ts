import { afterEach, expect, it, vi } from "vitest";
import { uploadPhotographerPhoto } from "./api-photographer";

afterEach(() => { vi.unstubAllGlobals(); });

it.each([
  ["NOT_FOUND", true],
  ["EVENT_NOT_FOUND", false],
] as const)("handles backend 404 code %s without hiding a missing event", async (code, fallback) => {
  const originalCrypto = globalThis.crypto;
  vi.stubGlobal("crypto", {
    subtle: {},
    getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto),
  });
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({
    success: false, errors: [{ code, message: "Resource not found" }],
  }), { status: 404 }));
  vi.stubGlobal("fetch", fetch);
  let transfers = 0;
  class MultipartXhr {
    status = 200;
    responseText = JSON.stringify({ success: true, data: { id: "legacy", status: "live" } });
    upload = { onprogress: null }; onload?: () => void;
    open() {}
    setRequestHeader() {}
    send() { transfers++; queueMicrotask(() => this.onload?.()); }
  }
  vi.stubGlobal("XMLHttpRequest", MultipartXhr);
  const work = uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) });
  if (fallback) expect(await work).toMatchObject({ id: "legacy" });
  else await expect(work).rejects.toMatchObject({ status: 404, errors: [{ code: "EVENT_NOT_FOUND" }] });
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(transfers).toBe(fallback ? 1 : 0);
});
