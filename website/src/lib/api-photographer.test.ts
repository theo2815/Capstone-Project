import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import { getEventMetadataVisibility, getMetadataSettings, OfflineUploadError, permanentDeletePhotographerPhotos, putEventMetadataVisibility, putMetadataSettings, restorePhotographerPhotos, trashPhotographerPhotos, uploadPhotographerPhoto } from "./api-photographer";
import { fetchPhotographerPayoutReports } from "./api-photographer-earnings";
import { report } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";

const { get, post, put } = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/api")>()), api: { get, post, put }, refreshAccessToken: vi.fn() }));
beforeEach(() => { get.mockReset(); post.mockReset(); put.mockReset(); });
afterEach(() => { vi.restoreAllMocks(); });
const ids = (count: number) => Array.from({ length: count }, (_, i) => `photo-${i}`);

it("uses the typed account metadata routes and forwards GET cancellation", async () => {
  const signal = new AbortController().signal;
  get.mockResolvedValueOnce({ metadataPublicByDefault: false }).mockResolvedValueOnce({ metadataPublic: true });
  put.mockResolvedValueOnce({ metadataPublicByDefault: true }).mockResolvedValueOnce({ metadataPublic: false });

  expect(await getMetadataSettings(signal)).toEqual({ metadataPublicByDefault: false });
  expect(await putMetadataSettings({ metadataPublicByDefault: true })).toEqual({ metadataPublicByDefault: true });
  expect(await getEventMetadataVisibility("event/1", signal)).toEqual({ metadataPublic: true });
  expect(await putEventMetadataVisibility("event/1", { metadataPublic: false })).toEqual({ metadataPublic: false });
  expect(get.mock.calls).toEqual([
    ["/me/photographer/metadata-settings", { signal }],
    ["/me/photographer/events/event%2F1/metadata-visibility", { signal }],
  ]);
  expect(put.mock.calls).toEqual([
    ["/me/photographer/metadata-settings", { metadataPublicByDefault: true }],
    ["/me/photographer/events/event%2F1/metadata-visibility", { metadataPublic: false }],
  ]);
});

it("unwraps the paginated photographer report response for report consumers", async () => {
  get.mockResolvedValue({ items: [report], total: 1, offset: 0, limit: 50 });
  expect(await fetchPhotographerPayoutReports()).toEqual([report]);
});

it("batches all selected pages within the 500-photo mutation limit", async () => {
  post.mockImplementation(async (_path, body) => ({ affected: body.photoIds.length }));
  const result = await trashPhotographerPhotos("event", ids(1001));
  expect(post.mock.calls.map(([, body]) => body.photoIds.length)).toEqual([500, 500, 1]);
  expect(result).toEqual({ affected: 1001, failed: [] });
});

it("keeps completed restore counts and identifies a failed batch for retry", async () => {
  post.mockResolvedValueOnce({ affected: 500 }).mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce({ affected: 1 });
  const result = await restorePhotographerPhotos("event", ids(1001));
  expect(result).toEqual({ affected: 501, failed: ids(1001).slice(500, 1000) });
});

it("combines deletion results, purchase holds, and failed requests without losing IDs", async () => {
  const selected = ids(251);
  post.mockResolvedValueOnce({ deleted: selected.slice(0, 90), blockedSold: selected.slice(90, 100) })
    .mockRejectedValueOnce(new Error("timeout"))
    .mockResolvedValueOnce({ deleted: selected.slice(200, 250), blockedSold: [], blockedPending: [selected[250]] });
  const result = await permanentDeletePhotographerPhotos("event", selected);
  expect(post.mock.calls.map(([, body]) => body.photoIds.length)).toEqual([100, 100, 51]);
  expect(result.deleted).toEqual([...selected.slice(0, 90), ...selected.slice(200, 250)]);
  expect(result.blockedSold).toEqual(selected.slice(90, 100));
  expect(result.blockedPending).toEqual([selected[250]]);
  expect(result.failed).toEqual(selected.slice(100, 200));
});

it("begins, sends raw bytes without credentials, then confirms the backend receipt", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  const original = globalThis.XMLHttpRequest;
  const sent: { method?: string; body?: Document | XMLHttpRequestBodyInit | null; headers: Record<string, string>; credentials?: boolean } = { headers: {} };
  class UploadXhr {
    status = 200; responseText = ""; withCredentials = true; upload = { onprogress: null };
    onload?: () => void;
    open(method: string) { sent.method = method; }
    setRequestHeader(name: string, value: string) { sent.headers[name] = value; }
    send(body?: Document | XMLHttpRequestBodyInit | null) { sent.body = body; sent.credentials = this.withCredentials; queueMicrotask(() => this.onload?.()); }
  }
  vi.stubGlobal("XMLHttpRequest", UploadXhr);
  const file = new File(["photo"], "race.jpg", { type: "image/jpeg" });
  const committed = { id: "photo-1", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockResolvedValueOnce({ mode: "direct", photoId: "photo-1", key: "staging/key", uploadUrl: "https://uploads.example.test/key?signature=1", expiresAt: "2099-01-01T00:00:00Z", intentExpiresAt: "2099-01-01T01:00:00Z", uploadHeaders: { "Content-Type": "image/jpeg" } }).mockResolvedValueOnce(committed);
  try {
    const result = await uploadPhotographerPhoto("event-1", file, undefined, { contentHash: "a".repeat(64) });
    expect(post.mock.calls.map(([path]) => path)).toEqual(["/me/photographer/events/event-1/photos/direct", "/me/photographer/events/event-1/photos/direct/commit"]);
    expect(post.mock.calls[0][1]).toMatchObject({ originalFilename: "race.jpg" });
    expect(post.mock.calls[1][1]).not.toHaveProperty("originalFilename");
    expect(sent.method).toBe("PUT");
    expect(sent.body).toBe(file);
    expect(sent.headers).toEqual({ "Content-Type": "image/jpeg" });
    expect(sent.credentials).toBe(false);
    expect(result).toEqual(committed);
  } finally { vi.stubGlobal("XMLHttpRequest", original); vi.unstubAllEnvs(); }
});

it("confirms an ambiguous PUT before sending the bytes again", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  const original = globalThis.XMLHttpRequest;
  let sends = 0;
  class LostReplyXhr {
    withCredentials = true; upload = { onprogress: null };
    onerror?: () => void;
    open() {}
    setRequestHeader() {}
    send() { sends++; queueMicrotask(() => this.onerror?.()); }
    abort() {}
  }
  vi.stubGlobal("XMLHttpRequest", LostReplyXhr);
  const receipt = { id: "saved", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockResolvedValueOnce({ mode: "direct", photoId: "saved", key: "slot", uploadUrl: "https://uploads.example.test/slot", uploadHeaders: {} }).mockResolvedValueOnce(receipt);
  try {
    expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) })).toEqual(receipt);
    expect(sends).toBe(1);
    expect(post).toHaveBeenCalledTimes(2);
  } finally { vi.stubGlobal("XMLHttpRequest", original); vi.unstubAllEnvs(); }
});

it("reuses the saved receipt first and prevents a new-session transfer", async () => {
  const receipt = { id: "saved", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockResolvedValueOnce(receipt);
  const file = new File(["a"], "a.jpg", { type: "image/jpeg" });
  expect(await uploadPhotographerPhoto("event", file, undefined, { contentHash: "a".repeat(64), checkpoint: { requestId: "request", photoId: "saved", key: "slot" } })).toEqual(receipt);
  expect(post.mock.calls.map(([path]) => path)).toEqual(["/me/photographer/events/event/photos/direct/commit"]);

  let release!: (value: unknown) => void;
  post.mockReset().mockImplementation(() => new Promise((resolve) => { release = resolve; }));
  const work = uploadPhotographerPhoto("event", file, undefined, { contentHash: "a".repeat(64) });
  useAuthStore.getState().logout();
  release({ mode: "direct", photoId: "saved", key: "slot", uploadUrl: "https://uploads.example.test/slot" });
  await expect(work).rejects.toThrow("Session changed");
  expect(post).toHaveBeenCalledTimes(1);
});

it("does not retransmit when confirmation remains ambiguous after a lost PUT reply", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  const original = globalThis.XMLHttpRequest;
  const nativeTimer = globalThis.setTimeout;
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) =>
    ms && ms < 5000 ? nativeTimer(callback, 0, ...args) : nativeTimer(callback, ms, ...args)) as typeof setTimeout);
  let sends = 0;
  class LostReplyXhr {
    upload = { onprogress: null }; onerror?: () => void;
    open() {}
    setRequestHeader() {}
    send() { sends++; queueMicrotask(() => this.onerror?.()); }
    abort() {}
  }
  vi.stubGlobal("XMLHttpRequest", LostReplyXhr);
  post.mockResolvedValueOnce({ mode: "direct", photoId: "p", key: "k", uploadUrl: "https://uploads.example.test/k" });
  post.mockRejectedValue(new ApiError([{ code: "UNKNOWN", message: "Gateway lost confirmation" }], 503));
  try {
    await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) })).rejects.toThrow("Gateway lost confirmation");
    expect(sends).toBe(1);
    expect(post).toHaveBeenCalledTimes(5);
  } finally { vi.stubGlobal("XMLHttpRequest", original); vi.unstubAllEnvs(); }
});

it("returns an existing receipt without a transfer", async () => {
  const receipt = { id: "already", status: "live", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockResolvedValueOnce({ mode: "existing", existing: receipt });
  const result = await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) });
  expect(result).toEqual(receipt);
  expect(post).toHaveBeenCalledTimes(1);
});

it("preserves backend error codes from a multipart fallback", async () => {
  const original = globalThis.XMLHttpRequest;
  class MultipartXhr {
    status = 409; responseText = JSON.stringify({ success: false, errors: [{ code: "PHOTO_DUPLICATE_DIFFERENT_EVENT", message: "Other event" }] });
    upload = { onprogress: null }; onload?: () => void;
    open() {}
    setRequestHeader() {}
    send() { queueMicrotask(() => this.onload?.()); }
  }
  vi.stubGlobal("XMLHttpRequest", MultipartXhr);
  post.mockResolvedValueOnce({ mode: "multipart" });
  try {
    await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) }))
      .rejects.toMatchObject({ status: 409, errors: [{ code: "PHOTO_DUPLICATE_DIFFERENT_EVENT" }] });
  } finally { vi.stubGlobal("XMLHttpRequest", original); }
});

it("uses multipart before a direct grant when Web Crypto is unavailable", async () => {
  const originalXhr = globalThis.XMLHttpRequest;
  const originalCrypto = globalThis.crypto;
  class MultipartXhr {
    status = 200; responseText = JSON.stringify({ success: true, data: { id: "multipart", status: "live" } });
    upload = { onprogress: null }; onload?: () => void;
    open() {}
    setRequestHeader() {}
    send() { queueMicrotask(() => this.onload?.()); }
  }
  vi.stubGlobal("XMLHttpRequest", MultipartXhr);
  vi.stubGlobal("crypto", undefined);
  try {
    expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) })).toMatchObject({ id: "multipart" });
    expect(post).not.toHaveBeenCalled();
  } finally { vi.stubGlobal("XMLHttpRequest", originalXhr); vi.stubGlobal("crypto", originalCrypto); }
});

it("uses multipart if SHA-256 is unavailable even with a supplied hash", async () => {
  const originalXhr = globalThis.XMLHttpRequest;
  const originalCrypto = globalThis.crypto;
  class MultipartXhr {
    status = 200; responseText = JSON.stringify({ success: true, data: { id: "multipart", status: "live" } });
    upload = { onprogress: null }; onload?: () => void;
    open() {}
    setRequestHeader() {}
    send() { queueMicrotask(() => this.onload?.()); }
  }
  vi.stubGlobal("XMLHttpRequest", MultipartXhr);
  vi.stubGlobal("crypto", { getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto) });
  try {
    expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) })).toMatchObject({ id: "multipart" });
    expect(post).not.toHaveBeenCalled();
  } finally { vi.stubGlobal("XMLHttpRequest", originalXhr); vi.stubGlobal("crypto", originalCrypto); }
});

it("replays the same request id after an ambiguous begin", async () => {
  const nativeTimer = globalThis.setTimeout;
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) =>
    nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args)) as typeof setTimeout);
  const receipt = { id: "already", status: "live", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockRejectedValueOnce(new TypeError("Connection lost")).mockResolvedValueOnce({ mode: "existing", existing: receipt });
  const checkpoints: { requestId: string }[] = [];
  expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, {
    contentHash: "a".repeat(64), onCheckpoint: (value) => checkpoints.push(value),
  })).toEqual(receipt);
  expect(post).toHaveBeenCalledTimes(2);
  expect(post.mock.calls[0][1].requestId).toBe(post.mock.calls[1][1].requestId);
  expect(checkpoints[0].requestId).toBe(post.mock.calls[0][1].requestId);
});

it("waits for Retry-After on verification contention and bounded 429s", async () => {
  const nativeTimer = globalThis.setTimeout;
  const delays: number[] = [];
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) => {
    delays.push(ms ?? 0);
    return nativeTimer(callback, ms && ms < 10000 ? 0 : ms, ...args);
  }) as typeof setTimeout);
  const receipt = { id: "saved", status: "live", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockRejectedValueOnce(new ApiError([{ code: "UPLOAD_IN_PROGRESS", message: "Verifying" }], 409, 5)).mockResolvedValueOnce(receipt);
  const file = new File(["a"], "a.jpg", { type: "image/jpeg" });
  const checkpoint = { requestId: "request", photoId: "saved", key: "slot" };
  expect(await uploadPhotographerPhoto("event", file, undefined, { contentHash: "a".repeat(64), checkpoint })).toEqual(receipt);
  expect(delays).toContain(5000);
  post.mockReset().mockRejectedValueOnce(new ApiError([{ code: "RATE_LIMITED", message: "Busy" }], 429, 5))
    .mockRejectedValueOnce(new ApiError([{ code: "RATE_LIMITED", message: "Busy" }], 429, 5))
    .mockResolvedValueOnce(receipt);
  expect(await uploadPhotographerPhoto("event", file, undefined, { contentHash: "a".repeat(64), checkpoint })).toEqual(receipt);
  expect(post).toHaveBeenCalledTimes(3);
  expect(delays.filter((delay) => delay === 5000)).toHaveLength(3);
  post.mockReset().mockRejectedValue(new ApiError([{ code: "RATE_LIMITED", message: "Busy" }], 429, 5));
  await expect(uploadPhotographerPhoto("event", file, undefined, { contentHash: "a".repeat(64), checkpoint })).rejects.toMatchObject({ status: 429 });
  expect(post).toHaveBeenCalledTimes(4);
});

it("renews a lost begin request only after its reservation is confirmed missing", async () => {
  let online = true;
  vi.spyOn(window.navigator, "onLine", "get").mockImplementation(() => online);
  const missing = new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Expired" }], 422);
  const receipt = { id: "saved", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  const checkpoints: { requestId: string }[] = [];
  const retryState = {};
  const file = new File(["a"], "a.jpg", { type: "image/jpeg" });
  post.mockImplementationOnce(async () => { online = false; throw new TypeError("Lost reply"); })
    .mockRejectedValueOnce(missing)
    .mockResolvedValueOnce({ mode: "existing", existing: receipt });
  const options = { contentHash: "a".repeat(64), retryState, onCheckpoint: (value: { requestId: string }) => checkpoints.push(value) };
  await expect(uploadPhotographerPhoto("event", file, undefined, options)).rejects.toBeInstanceOf(OfflineUploadError);
  online = true;
  expect(await uploadPhotographerPhoto("event", file, undefined, { ...options, checkpoint: checkpoints.at(-1) })).toEqual(receipt);
  expect(post).toHaveBeenCalledTimes(3);
  expect(post.mock.calls[0][1].requestId).toBe(post.mock.calls[1][1].requestId);
  expect(post.mock.calls[2][1].requestId).not.toBe(post.mock.calls[1][1].requestId);
});

it("reconciles an R2 XML 403 before renewing a confirmed-missing expired grant", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  vi.spyOn(Math, "random").mockReturnValue(0.5);
  const nativeTimer = globalThis.setTimeout;
  const delays: number[] = [];
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) => {
    if (ms && ms < 5000) delays.push(ms);
    return nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args);
  }) as typeof setTimeout);
  let now = Date.now();
  vi.spyOn(Date, "now").mockImplementation(() => now);
  const firstExpiry = new Date(now + 60_000).toISOString();
  const sent: { url: string; headers: Record<string, string>; credentials: boolean }[] = [];
  class StorageXhr {
    status = 200;
    responseText = "<Error>ExpiredToken</Error>";
    withCredentials = true;
    upload = { onprogress: null };
    onload?: () => void;
    private url = "";
    private headers: Record<string, string> = {};
    open(_method: string, url: string) { this.url = url; }
    setRequestHeader(name: string, value: string) { this.headers[name] = value; }
    send() {
      sent.push({ url: this.url, headers: this.headers, credentials: this.withCredentials });
      if (sent.length === 1) { this.status = 403; now += 120_000; }
      queueMicrotask(() => this.onload?.());
    }
    abort() {}
  }
  vi.stubGlobal("XMLHttpRequest", StorageXhr);
  const missing = new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Missing" }], 422);
  const receipt = { id: "saved", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockResolvedValueOnce({ mode: "direct", photoId: "old", key: "old-key", uploadUrl: "https://uploads.example.test/old?sig=1", expiresAt: firstExpiry, uploadHeaders: { "Content-Type": "image/jpeg" } })
    .mockRejectedValueOnce(missing)
    .mockResolvedValueOnce({ mode: "direct", photoId: "new", key: "new-key", uploadUrl: "https://uploads.example.test/new?sig=2", expiresAt: new Date(now + 300_000).toISOString(), uploadHeaders: { "Content-Type": "image/jpeg" } })
    .mockResolvedValueOnce(receipt);
  try {
    expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) })).toEqual(receipt);
    expect(post.mock.calls.map(([path]) => path)).toEqual([
      "/me/photographer/events/event/photos/direct", "/me/photographer/events/event/photos/direct/commit",
      "/me/photographer/events/event/photos/direct", "/me/photographer/events/event/photos/direct/commit",
    ]);
    expect(post.mock.calls[2][1].requestId).not.toBe(post.mock.calls[0][1].requestId);
    expect(sent).toEqual([
      { url: "https://uploads.example.test/old?sig=1", headers: { "Content-Type": "image/jpeg" }, credentials: false },
      { url: "https://uploads.example.test/new?sig=2", headers: { "Content-Type": "image/jpeg" }, credentials: false },
    ]);
    expect(delays).toContain(1000);
  } finally { vi.unstubAllGlobals(); vi.unstubAllEnvs(); }
});

it("shares begin retries between transient failures and confirmed-missing renewal", async () => {
  vi.spyOn(Math, "random").mockReturnValue(0.5);
  const nativeTimer = globalThis.setTimeout;
  const delays: number[] = [];
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) => {
    if (ms && ms < 5000) delays.push(ms);
    return nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args);
  }) as typeof setTimeout);
  const missing = new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Expired" }], 422);
  const retryState: { beginRetries?: number } = {};
  post.mockRejectedValueOnce(new TypeError("Lost begin reply"))
    .mockRejectedValueOnce(new TypeError("Lost begin reply"))
    .mockRejectedValueOnce(new TypeError("Lost begin reply"))
    .mockRejectedValue(missing);
  await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, {
    contentHash: "a".repeat(64), retryState,
  })).rejects.toBe(missing);
  expect(post).toHaveBeenCalledTimes(4);
  expect(new Set(post.mock.calls.map(([, body]) => body.requestId)).size).toBe(1);
  expect(retryState.beginRetries).toBe(3);
  expect(delays).toEqual([1000, 2000, 4000]);
  post.mockReset().mockRejectedValue(new ApiError([{ code: "UPLOAD_REQUEST_CONFLICT", message: "Conflict" }], 409));
  await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, {
    contentHash: "a".repeat(64),
  })).rejects.toMatchObject({ status: 409, errors: [{ code: "UPLOAD_REQUEST_CONFLICT" }] });
  expect(post).toHaveBeenCalledTimes(1);
});

it("delays a confirmed-missing begin renewal using its begin retry budget", async () => {
  vi.spyOn(Math, "random").mockReturnValue(0.5);
  const nativeTimer = globalThis.setTimeout;
  const delays: number[] = [];
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) => {
    if (ms && ms < 5000) delays.push(ms);
    return nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args);
  }) as typeof setTimeout);
  const receipt = { id: "saved", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockRejectedValueOnce(new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Expired" }], 422, 3))
    .mockResolvedValueOnce({ mode: "existing", existing: receipt });
  const retryState: { beginRetries?: number } = {};
  expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, {
    contentHash: "a".repeat(64), retryState,
  })).toEqual(receipt);
  expect(retryState.beginRetries).toBe(1);
  expect(delays).toContain(3000);
  expect(post.mock.calls[1][1].requestId).not.toBe(post.mock.calls[0][1].requestId);
});

it("shares transfer retries between in-call failures and expired-storage renewals", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  vi.spyOn(Math, "random").mockReturnValue(0.5);
  const nativeTimer = globalThis.setTimeout;
  const delays: number[] = [];
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) => {
    if (ms && ms < 5000) delays.push(ms);
    return nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args);
  }) as typeof setTimeout);
  let now = Date.now();
  vi.spyOn(Date, "now").mockImplementation(() => now);
  let puts = 0;
  class FailingStorageXhr {
    status = 403;
    upload = { onprogress: null };
    onload?: () => void;
    onerror?: () => void;
    open() {}
    setRequestHeader() {}
    send() {
      puts++;
      if (puts > 1) now += 20_000;
      queueMicrotask(() => puts === 1 ? this.onerror?.() : this.onload?.());
    }
    abort() {}
  }
  vi.stubGlobal("XMLHttpRequest", FailingStorageXhr);
  const missing = new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Missing" }], 422);
  let begins = 0;
  let commits = 0;
  post.mockImplementation(async (path: string) => {
    if (path.endsWith("/direct/commit")) { commits++; throw missing; }
    begins++;
    return { mode: "direct", photoId: `photo-${begins}`, key: `key-${begins}`, uploadUrl: `https://uploads.example.test/${begins}`, expiresAt: new Date(now + 10_000).toISOString() };
  });
  const retryState: { transferRetries?: number } = {};
  try {
    await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, {
      contentHash: "a".repeat(64), retryState,
    })).rejects.toMatchObject({ status: 403, errors: [{ code: "UPLOAD_TRANSFER_FAILED" }] });
    expect(puts).toBe(4);
    expect(begins).toBe(3);
    expect(commits).toBe(4);
    expect(retryState.transferRetries).toBe(3);
    expect(delays).toEqual([1000, 2000, 4000]);
  } finally { vi.unstubAllGlobals(); vi.unstubAllEnvs(); }
});

it("does not renew a storage 403 while its grant is unexpired", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  class DeniedXhr {
    status = 403; upload = { onprogress: null }; onload?: () => void;
    open() {}
    setRequestHeader() {}
    send() { queueMicrotask(() => this.onload?.()); }
  }
  vi.stubGlobal("XMLHttpRequest", DeniedXhr);
  post.mockResolvedValueOnce({ mode: "direct", photoId: "old", key: "key", uploadUrl: "https://uploads.example.test/key", expiresAt: "2099-01-01T00:00:00Z" })
    .mockRejectedValueOnce(new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Missing" }], 422));
  try {
    await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) }))
      .rejects.toMatchObject({ status: 403, errors: [{ code: "UPLOAD_TRANSFER_FAILED" }] });
    expect(post.mock.calls.map(([path]) => path)).toEqual([
      "/me/photographer/events/event/photos/direct", "/me/photographer/events/event/photos/direct/commit",
    ]);
  } finally { vi.unstubAllGlobals(); vi.unstubAllEnvs(); }
});

it("keeps a backend 403 terminal before any storage transfer or multipart fallback", async () => {
  const send = vi.fn();
  vi.stubGlobal("XMLHttpRequest", class { upload = { onprogress: null }; open() {} setRequestHeader() {} send = send; });
  post.mockRejectedValueOnce(new ApiError([{ code: "PHOTOGRAPHER_NOT_VERIFIED", message: "Verification required" }], 403));
  try {
    await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) }))
      .rejects.toMatchObject({ status: 403, errors: [{ code: "PHOTOGRAPHER_NOT_VERIFIED" }] });
    expect(post).toHaveBeenCalledTimes(1);
    expect(send).not.toHaveBeenCalled();
  } finally { vi.unstubAllGlobals(); }
});

it("rejects an unexpected direct-upload origin before sending any bytes", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  const send = vi.fn();
  vi.stubGlobal("XMLHttpRequest", class { upload = { onprogress: null }; open() {} setRequestHeader() {} send = send; });
  post.mockResolvedValueOnce({ mode: "direct", photoId: "photo", key: "key", uploadUrl: "https://elsewhere.example.test/key" });
  try {
    await expect(uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) }))
      .rejects.toThrow("Invalid upload URL");
    expect(send).not.toHaveBeenCalled();
    expect(post).toHaveBeenCalledTimes(1);
  } finally { vi.unstubAllGlobals(); vi.unstubAllEnvs(); }
});

it("settles a raw PUT after 120 seconds without byte progress", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  let aborted = false;
  class StalledXhr {
    upload = { onprogress: null };
    open() {}
    setRequestHeader() {}
    send() {}
    abort() { aborted = true; }
  }
  vi.stubGlobal("XMLHttpRequest", StalledXhr);
  const receipt = { id: "saved", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockResolvedValueOnce({ mode: "direct", photoId: "saved", key: "key", uploadUrl: "https://uploads.example.test/key" })
    .mockResolvedValueOnce(receipt);
  try {
    const work = uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) });
    await vi.advanceTimersByTimeAsync(120_000);
    expect(await work).toEqual(receipt);
    expect(aborted).toBe(true);
    expect(post).toHaveBeenCalledTimes(2);
  } finally { vi.useRealTimers(); vi.unstubAllGlobals(); vi.unstubAllEnvs(); }
});

it("starts a fresh request only after an expired grant is confirmed missing", async () => {
  const old = { requestId: "old-request", photoId: "old-photo", key: "old-key", putExpiresAt: "2000-01-01T00:00:00Z" };
  const receipt = { id: "new-photo", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockRejectedValueOnce(new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Missing" }], 422))
    .mockResolvedValueOnce({ mode: "existing", existing: receipt });
  const saved: { requestId: string }[] = [];
  expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, {
    contentHash: "a".repeat(64), checkpoint: old, onCheckpoint: (value) => saved.push(value),
  })).toEqual(receipt);
  expect(post.mock.calls[1][1].requestId).not.toBe(old.requestId);
  expect(saved[0].requestId).toBe(post.mock.calls[1][1].requestId);
});

it("lets the backend expire a grant when the browser clock is behind", async () => {
  vi.spyOn(Date, "now").mockReturnValue(Date.parse("1990-01-01T00:00:00Z"));
  const nativeTimer = globalThis.setTimeout;
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) =>
    nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args)) as typeof setTimeout);
  const checkpoint = { requestId: "old-request", photoId: "old-photo", key: "old-key", putExpiresAt: "2000-01-01T00:00:00Z" };
  const missing = new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Missing" }], 422);
  const receipt = { id: "new-photo", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" };
  post.mockRejectedValueOnce(missing).mockRejectedValueOnce(missing)
    .mockResolvedValueOnce({ mode: "existing", existing: receipt });

  expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, {
    contentHash: "a".repeat(64), checkpoint,
  })).toEqual(receipt);
  expect(post.mock.calls.map(([path]) => path)).toEqual([
    "/me/photographer/events/event/photos/direct/commit",
    "/me/photographer/events/event/photos/direct",
    "/me/photographer/events/event/photos/direct",
  ]);
  expect(post.mock.calls[1][1].requestId).toBe(checkpoint.requestId);
  expect(post.mock.calls[2][1].requestId).not.toBe(checkpoint.requestId);
});

it("reports multipart transfer progress and confirmation phases", async () => {
  const original = globalThis.XMLHttpRequest;
  class MultipartXhr {
    status = 200; responseText = JSON.stringify({ success: true, data: { id: "multipart", status: "processing" } });
    upload: { onprogress: ((event: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = { onprogress: null };
    onload?: () => void;
    open() {}
    setRequestHeader() {}
    send() {
      this.upload.onprogress?.({ lengthComputable: true, loaded: 1, total: 1 });
      queueMicrotask(() => this.onload?.());
    }
  }
  vi.stubGlobal("XMLHttpRequest", MultipartXhr);
  post.mockResolvedValueOnce({ mode: "multipart" });
  const phases: string[] = [];
  const progress: number[] = [];
  try {
    expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), (event) => progress.push(event.percent), {
      contentHash: "a".repeat(64), onPhase: (phase) => phases.push(phase),
    })).toMatchObject({ id: "multipart", status: "processing" });
    expect(progress).toEqual([100]);
    expect(phases).toEqual(["preparing", "uploading", "confirming"]);
  } finally { vi.stubGlobal("XMLHttpRequest", original); }
});

it("keeps the fifteen-minute transfer cap across PUT retries", async () => {
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  const original = globalThis.XMLHttpRequest;
  const nativeTimer = globalThis.setTimeout;
  const caps: number[] = [];
  let clock = 1_000_000;
  vi.spyOn(Date, "now").mockImplementation(() => clock);
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) => {
    if (ms && ms >= 800_000) caps.push(ms);
    return nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args);
  }) as typeof setTimeout);
  let sends = 0;
  class RetryXhr {
    status = 200; upload = { onprogress: null }; onload?: () => void; onerror?: () => void;
    open() {}
    setRequestHeader() {}
    send() {
      sends++;
      if (sends === 1) {
        clock += 60_000;
        queueMicrotask(() => this.onerror?.());
      } else queueMicrotask(() => this.onload?.());
    }
    abort() {}
  }
  vi.stubGlobal("XMLHttpRequest", RetryXhr);
  post.mockResolvedValueOnce({ mode: "direct", photoId: "p", key: "k", uploadUrl: "https://uploads.example.test/k" })
    .mockRejectedValueOnce(new ApiError([{ code: "UPLOAD_OBJECT_MISSING", message: "Missing" }], 422))
    .mockResolvedValueOnce({ id: "p", status: "live" });
  try {
    await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) });
    expect(sends).toBe(2);
    expect(caps).toEqual([900_000, 840_000]);
  } finally { vi.stubGlobal("XMLHttpRequest", original); vi.unstubAllEnvs(); }
});

it("retries a failed multipart transfer but keeps terminal errors terminal", async () => {
  const nativeTimer = globalThis.setTimeout;
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) =>
    nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args)) as typeof setTimeout);
  let sends = 0;
  class MultipartXhr {
    status = 200;
    responseText = JSON.stringify({ success: true, data: { id: "saved", status: "processing" } });
    upload = { onprogress: null };
    onload?: () => void;
    onerror?: () => void;
    open() {}
    setRequestHeader() {}
    send() { sends++; queueMicrotask(() => sends === 1 ? this.onerror?.() : this.onload?.()); }
    abort() {}
  }
  vi.stubGlobal("XMLHttpRequest", MultipartXhr);
  post.mockResolvedValue({ mode: "multipart" });
  expect(await uploadPhotographerPhoto("event", new File(["a"], "a.jpg", { type: "image/jpeg" }), undefined, { contentHash: "a".repeat(64) })).toMatchObject({ id: "saved" });
  expect(sends).toBe(2);
  vi.unstubAllGlobals();
});

it("retains begin retry budget across automatic offline recovery", async () => {
  const nativeTimer = globalThis.setTimeout;
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) =>
    nativeTimer(callback, ms && ms < 5000 ? 0 : ms, ...args)) as typeof setTimeout);
  const state = {};
  let online = true;
  vi.spyOn(window.navigator, "onLine", "get").mockImplementation(() => online);
  let calls = 0;
  post.mockImplementation(() => {
    calls++;
    if (calls <= 2) online = false;
    return Promise.reject(new TypeError("Connection lost"));
  });
  const file = new File(["a"], "a.jpg", { type: "image/jpeg" });
  const options = { contentHash: "a".repeat(64), retryState: state, checkpoint: { requestId: "same" } };
  await expect(uploadPhotographerPhoto("event", file, undefined, options)).rejects.toBeInstanceOf(OfflineUploadError);
  online = true;
  await expect(uploadPhotographerPhoto("event", file, undefined, options)).rejects.toBeInstanceOf(OfflineUploadError);
  online = true;
  await expect(uploadPhotographerPhoto("event", file, undefined, options)).rejects.toThrow("Connection lost");
  expect(calls).toBe(4);
});

it("parks multipart after a connection loss and resumes with the same retry budget", async () => {
  let online = true;
  vi.spyOn(window.navigator, "onLine", "get").mockImplementation(() => online);
  let sends = 0;
  class MultipartXhr {
    status = 200;
    responseText = JSON.stringify({ success: true, data: { id: "saved", status: "processing" } });
    upload = { onprogress: null };
    onload?: () => void;
    onerror?: () => void;
    open() {}
    setRequestHeader() {}
    send() {
      sends++;
      queueMicrotask(() => {
        if (sends === 1) { online = false; this.onerror?.(); }
        else this.onload?.();
      });
    }
    abort() {}
  }
  vi.stubGlobal("XMLHttpRequest", MultipartXhr);
  post.mockResolvedValue({ mode: "multipart" });
  const retryState = {};
  const options = { contentHash: "a".repeat(64), retryState };
  const file = new File(["a"], "a.jpg", { type: "image/jpeg" });
  await expect(uploadPhotographerPhoto("event", file, undefined, options)).rejects.toBeInstanceOf(OfflineUploadError);
  expect(retryState).toMatchObject({ transferRetries: 1 });
  online = true;
  expect(await uploadPhotographerPhoto("event", file, undefined, options)).toMatchObject({ id: "saved" });
  expect(sends).toBe(2);
});
