import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import { OfflineUploadError, permanentDeletePhotographerPhotos, restorePhotographerPhotos, trashPhotographerPhotos, uploadPhotographerPhoto } from "./api-photographer";
import { fetchPhotographerPayoutReports } from "./api-photographer-earnings";
import { report } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";

const { get, post } = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/api")>()), api: { get, post }, refreshAccessToken: vi.fn() }));
beforeEach(() => { get.mockReset(); post.mockReset(); });
afterEach(() => { vi.restoreAllMocks(); });
const ids = (count: number) => Array.from({ length: count }, (_, i) => `photo-${i}`);

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

it("waits for UPLOAD_IN_PROGRESS Retry-After and does not retry a 429", async () => {
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
  post.mockReset().mockRejectedValueOnce(new ApiError([{ code: "RATE_LIMITED", message: "Slow down" }], 429, 5));
  await expect(uploadPhotographerPhoto("event", file, undefined, { contentHash: "a".repeat(64), checkpoint })).rejects.toMatchObject({ status: 429 });
  expect(post).toHaveBeenCalledTimes(1);
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
