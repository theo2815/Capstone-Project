import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, refreshAccessToken, setStepUpHandler } from "./api";
import { clearTokens, getAccessToken, getRefreshToken, setTokens } from "./auth";
import { useAuthStore } from "@/store/auth-store";
const user = { id: "runner", role: "RUNNER" as const, name: "Runner", email: "runner@example.test", createdAt: "2026-09-10" };
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}
function response(data: unknown, status = 200) {
  return new Response(JSON.stringify(status === 200 ? { success: true, data } : { success: false, errors: data }), { status });
}
function nextSession() {
  useAuthStore.getState().logout(); useAuthStore.getState().setUser(user);
  setTokens("new-access", "new-refresh");
}
beforeEach(() => { useAuthStore.getState().setUser(user); setTokens("old-access", "old-refresh"); });
afterEach(() => { useAuthStore.getState().logout(); clearTokens(); setStepUpHandler(null); vi.unstubAllGlobals(); });

it.each(["query cancellation", "request timeout"])("GET distinguishes %s when supplied a Query signal", async (cause) => {
  const query = new AbortController();
  const timeout = new AbortController();
  const timer = vi.spyOn(AbortSignal, "timeout").mockReturnValue(timeout.signal);
  vi.stubGlobal("fetch", vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => {
    options.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
  })));
  try {
    const request = api.get("/me/orders", { signal: query.signal });
    const result = expect(request).rejects.toMatchObject(cause === "query cancellation" ? { name: "AbortError" } : { errors: [{ code: "TIMEOUT" }] });
    (cause === "query cancellation" ? query : timeout).abort();
    await result;
    expect(timer).toHaveBeenCalledWith(30_000);
  } finally {
    timer.mockRestore();
  }
});

it("POST carries a custom timeout and caller signal through the refresh retry", async () => {
  const signal = new AbortController();
  const timeout = vi.spyOn(AbortSignal, "timeout");
  const fetch = vi.fn().mockResolvedValueOnce(response([], 401))
    .mockResolvedValueOnce(response({ accessToken: "fresh", refreshToken: "fresh-refresh" }))
    .mockResolvedValueOnce(response({ saved: true }));
  vi.stubGlobal("fetch", fetch);
  try {
    expect(await api.post("/me/photographer/events/e/photos/direct/commit", {}, { signal: signal.signal, timeoutMs: 120_000 })).toEqual({ saved: true });
    expect(timeout).toHaveBeenCalledWith(120_000);
    expect(fetch.mock.calls[2][1].signal).not.toBe(signal.signal);
    signal.abort();
    expect(fetch.mock.calls[2][1].signal.aborted).toBe(true);
  } finally { timeout.mockRestore(); }
});

it("deduplicates refresh within a session, but never joins or commits an old session's refresh", async () => {
  const old = deferred<Response>(), current = deferred<Response>();
  const fetch = vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise); vi.stubGlobal("fetch", fetch);
  const first = refreshAccessToken(); expect(refreshAccessToken()).toBe(first);
  nextSession();
  const second = refreshAccessToken(); expect(second).not.toBe(first);
  old.resolve(response({ accessToken: "stale", refreshToken: "stale-refresh" }));
  expect(await first).toBeNull();
  expect(getAccessToken()).toBe("new-access");
  expect(refreshAccessToken()).toBe(second); // An old finally must not release the new lock.
  current.resolve(response({ accessToken: "fresh", refreshToken: "fresh-refresh" }));
  expect(await second).toBe("fresh");
  expect(getRefreshToken()).toBe("fresh-refresh"); expect(fetch).toHaveBeenCalledTimes(2);
});

it("retains the same-session cross-tab rotation fallback", async () => {
  const pending = deferred<Response>();
  const fetch = vi.fn().mockReturnValueOnce(pending.promise).mockResolvedValueOnce(response({ accessToken: "rotated", refreshToken: "rotated-refresh" }));
  vi.stubGlobal("fetch", fetch);
  const request = refreshAccessToken(); setTokens("other-tab-access", "other-tab-refresh");
  pending.resolve(response([], 401));
  expect(await request).toBe("rotated");
  expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({ refreshToken: "other-tab-refresh" });
});

it.each(["before refresh", "during refresh"])("does not retry an old mutation or clear new credentials when the session ends %s", async (timing) => {
  const pending = deferred<Response>(), refresh = deferred<Response>();
  const fetch = vi.fn().mockReturnValueOnce(pending.promise).mockReturnValueOnce(refresh.promise); vi.stubGlobal("fetch", fetch);
  const result = api.post("/me/cart/items", { photoId: "old-photo" }).catch((e: Error) => e.message);
  if (timing === "before refresh") nextSession();
  pending.resolve(response([], 401));
  if (timing === "during refresh") {
    // Allow fetch + body parsing to reach the refresh call.
    for (let i = 0; i < 10; i++) await Promise.resolve();
    expect(fetch).toHaveBeenCalledTimes(2); nextSession(); refresh.resolve(response([], 401));
  }
  expect(await result).toBe("Session changed");
  expect(fetch).toHaveBeenCalledTimes(timing === "before refresh" ? 1 : 2);
  expect(getAccessToken()).toBe("new-access"); expect(getRefreshToken()).toBe("new-refresh");
});

it("discards an old step-up result without replaying the mutation as the new user", async () => {
  const proof = deferred<string | null>(); const prompted = vi.fn(() => proof.promise); setStepUpHandler(prompted);
  const fetch = vi.fn().mockResolvedValue(response([{ code: "STEP_UP_REQUIRED" }], 403)); vi.stubGlobal("fetch", fetch);
  const result = api.post("/me/photographer/payouts", {}).catch((e: Error) => e.message);
  for (let i = 0; i < 10; i++) await Promise.resolve();
  expect(prompted).toHaveBeenCalledTimes(1); nextSession(); proof.resolve("old-proof");
  expect(await result).toBe("Session changed"); expect(fetch).toHaveBeenCalledTimes(1);
});

it("preserves a caller's command key through current-session refresh and step-up retries", async () => {
  const fetch = vi.fn()
    .mockResolvedValueOnce(response([], 401))
    .mockResolvedValueOnce(response({ accessToken: "fresh", refreshToken: "fresh-refresh" }))
    .mockResolvedValueOnce(response([{ code: "STEP_UP_REQUIRED" }], 403))
    .mockResolvedValueOnce(response({ saved: true }));
  vi.stubGlobal("fetch", fetch); setStepUpHandler(async () => "proof");
  expect(await api.post("/admin/users/user-1/message", { subject: "Hello", body: "Message" }, { headers: { "Idempotency-Key": "caller-owned-key" } })).toEqual({ saved: true });
  for (const index of [0, 2, 3]) expect(fetch.mock.calls[index][1].headers).toMatchObject({ "Idempotency-Key": "caller-owned-key" });
  expect(fetch.mock.calls[3][1].headers).toMatchObject({ Authorization: "Bearer fresh", "X-Step-Up": "proof" });
});

it("GET Blob decodes private evidence bytes without forwarding its decoder option", async () => {
  const bytes = new Uint8Array([137, 80, 78, 71]);
  const fetch = vi.fn().mockResolvedValue(new Response(bytes, { headers: { "Content-Type": "image/png" } }));
  vi.stubGlobal("fetch", fetch);
  const blob = await api.getBlob("/admin/photo-removal-requests/case/evidence");
  expect(blob.type).toBe("image/png"); expect(new Uint8Array(await blob.arrayBuffer())).toEqual(bytes);
  expect(fetch.mock.calls[0][1]).not.toHaveProperty("responseType");
});

it("GET Blob preserves JSON error details and Retry-After", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ success: false, errors: [{ code: "FORBIDDEN", message: "Private evidence" }] }), { status: 403, headers: { "Retry-After": "42" } }));
  vi.stubGlobal("fetch", fetch);
  await expect(api.getBlob("/admin/photo-removal-requests/case/evidence")).rejects.toMatchObject({ status: 403, retryAfterSeconds: 42, errors: [{ code: "FORBIDDEN", message: "Private evidence" }] });
});

it("GET Blob retains refresh, step-up and the caller signal through retries", async () => {
  const caller = new AbortController();
  const fetch = vi.fn().mockResolvedValueOnce(response([], 401)).mockResolvedValueOnce(response({ accessToken: "fresh", refreshToken: "fresh-refresh" })).mockResolvedValueOnce(response([{ code: "STEP_UP_REQUIRED" }], 403)).mockResolvedValueOnce(new Response("evidence", { headers: { "Content-Type": "image/jpeg" } }));
  vi.stubGlobal("fetch", fetch); setStepUpHandler(async () => "proof");
  expect((await api.getBlob("/admin/photo-removal-requests/case/evidence", { signal: caller.signal })).type).toBe("image/jpeg");
  expect(fetch.mock.calls[3][1].headers).toMatchObject({ Authorization: "Bearer fresh", "X-Step-Up": "proof" });
  caller.abort(); expect(fetch.mock.calls[3][1].signal.aborted).toBe(true);
});

it.each(["query cancellation", "request timeout"])("GET Blob distinguishes %s", async (cause) => {
  const query = new AbortController(), timeout = new AbortController();
  const timer = vi.spyOn(AbortSignal, "timeout").mockReturnValue(timeout.signal);
  vi.stubGlobal("fetch", vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => { options.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))); })));
  try {
    const result = expect(api.getBlob("/admin/photo-removal-requests/case/evidence", { signal: query.signal })).rejects.toMatchObject(cause === "query cancellation" ? { name: "AbortError" } : { errors: [{ code: "TIMEOUT" }] });
    (cause === "query cancellation" ? query : timeout).abort(); await result;
  } finally { timer.mockRestore(); }
});

it.each(["query cancellation", "request timeout"])("GET Blob distinguishes %s while reading image bytes", async (cause) => {
  const query = new AbortController(), timeout = new AbortController();
  const timer = vi.spyOn(AbortSignal, "timeout").mockReturnValue(timeout.signal);
  let started!: () => void;
  const reading = new Promise<void>((resolve) => { started = resolve; });
  vi.stubGlobal("fetch", vi.fn(async (_url, options: RequestInit) => ({
    ok: true,
    blob: () => { started(); return new Promise((_resolve, reject) => { options.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))); }); },
  })));
  try {
    const result = expect(api.getBlob("/admin/photo-removal-requests/case/evidence", { signal: query.signal })).rejects.toMatchObject(cause === "query cancellation" ? { name: "AbortError" } : { errors: [{ code: "TIMEOUT" }] });
    await reading; (cause === "query cancellation" ? query : timeout).abort(); await result;
  } finally { timer.mockRestore(); }
});

it("an explicit sign-out reaches the login page without a redirect to the previous page", async () => {
  const location = { pathname: "/dashboard/events/event-1", search: "", hash: "", href: "" };
  vi.stubGlobal("location", location);
  vi.stubGlobal("fetch", vi.fn(async () => response([], 401)));
  useAuthStore.getState().logout();
  clearTokens();
  await expect(api.get("/me/photographer/events/event-1")).rejects.toThrow("Unauthorized");
  expect(location.href).toBe("/login");
});

it("a session that expired while signed in still redirects back to the page", async () => {
  const location = { pathname: "/dashboard/events/event-1", search: "", hash: "", href: "" };
  vi.stubGlobal("location", location);
  vi.stubGlobal("fetch", vi.fn(async () => response([], 401)));
  clearTokens();
  await expect(api.get("/me/photographer/events/event-1")).rejects.toThrow("Unauthorized");
  expect(location.href).toBe("/login?redirect=" + encodeURIComponent("/dashboard/events/event-1"));
});
