import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { webcrypto } from "node:crypto";
import { clearAdminActionIntents } from "@/lib/admin-action-intents";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAdminPhotographerView } from "@/lib/admin-photographer-view";
import { api, ApiError } from "@/lib/api";
import { admin, adminUser, adminUserSettings } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { adminUserKeys, useAdminUsers, useAdminUserActions, useAdminUserDetail, useAdminPhotographerSettings, usePendingVerificationsCount } from "./use-admin-users";

let client: QueryClient;
const listKey = adminUserKeys.list(admin.id);
const detailKey = adminUserKeys.detail(admin.id, adminUser.userId);
const settingsKey = adminUserKeys.settings(admin.id, adminUser.userId);
const detail = { ...adminUser, decisionLog: [] };
function wrapper({ children }: { children: ReactNode }) { return <QueryClientProvider client={client}>{children}</QueryClientProvider>; }
function deferred<T>() {
  let resolve!: (value: T) => void; let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto); clearAdminActionIntents();
  useAuthStore.getState().setUser(admin); useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(listKey, [adminUser]); client.setQueryData(detailKey, detail); client.setQueryData(settingsKey, adminUserSettings);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); clearAdminActionIntents(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });
const actions = ["approve", "reject", "suspend", "unsuspend", "resetVerification"] as const;
function invoke(actionsHook: ReturnType<typeof useAdminUserActions>, action: typeof actions[number]) {
  return action === "approve" ? actionsHook.approve(adminUser.userId, 7)
    : action === "unsuspend" ? actionsHook.unsuspend(adminUser.userId)
    : action === "suspend" ? actionsHook.suspend(adminUser.userId, "Please fix the watermark")
    : actionsHook[action](adminUser.userId, "Please fix the watermark", 7);
}
it("shares user claims across views and forwards the reviewed revision", async () => {
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValue(request.promise);
  const { result } = renderHook(() => [useAdminUserActions(), useAdminUserActions()], { wrapper });
  let first!: Promise<boolean>, blocked!: Promise<boolean>;
  act(() => { first = result.current[0].approve(adminUser.userId, 7); blocked = result.current[1].reject(adminUser.userId, "Fix", 7); });
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect(await blocked).toBe(false);
  expect(post).toHaveBeenCalledWith(`/admin/users/${adminUser.userId}/approve`, { expectedRevision: 7 });
  await act(async () => { request.resolve(adminUser); expect(await first).toBe(true); });
});
it("names the action in plain words when the request never reaches the server", async () => {
  vi.spyOn(api, "post").mockRejectedValue(new TypeError("Failed to fetch"));
  const { result } = renderHook(() => useAdminUserActions(), { wrapper });
  await act(async () => { expect(await result.current.approve(adminUser.userId, 7)).toBe(false); });
  expect(useToastStore.getState().toasts.at(-1)?.message).toBe("Couldn't approve — please try again.");
});
it("reserves a whole verification batch and refreshes once while retaining partial failures", async () => {
  const request = deferred<unknown>();
  const post = vi.spyOn(api, "post").mockImplementation((path) => path.includes("second") ? Promise.reject(new Error("Refused")) : request.promise);
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const { result } = renderHook(() => [useAdminUserActions(), useAdminUserActions()], { wrapper });
  let batch!: Promise<boolean[] | null>;
  act(() => { batch = result.current[0].verifyBatch([{ userId: adminUser.userId, verificationRevision: 7 }, { userId: "second", verificationRevision: 9 }], "reject", "Fix"); });
  await expect(result.current[1].approve("second", 9)).resolves.toBe(false);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect(invalidate).not.toHaveBeenCalled();
  await act(async () => { request.resolve(adminUser); expect(await batch).toEqual([true, false]); });
  expect(invalidate).toHaveBeenCalledTimes(2);
});
it("does not turn an accepted user action into failure when refresh rejects", async () => {
  vi.spyOn(api, "post").mockResolvedValue(adminUser);
  vi.spyOn(client, "invalidateQueries").mockRejectedValue(new Error("Refresh unavailable"));
  const { result } = renderHook(useAdminUserActions, { wrapper });
  await expect(result.current.approve(adminUser.userId, 7)).resolves.toBe(true);
  expect(client.getQueryData(listKey)).toEqual([adminUser]);
  expect(useToastStore.getState().toasts.some((toast) => toast.kind === "error")).toBe(false);
});
it("messages share the user guard and retry the same key until confirmed, then allow another intentional send", async () => {
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
  const { result } = renderHook(useAdminUserActions, { wrapper });
  let pending!: Promise<boolean>;
  act(() => { pending = result.current.message(adminUser.userId, { subject: "Hello", body: "Private" }); });
  await expect(result.current.approve(adminUser.userId, 7)).resolves.toBe(false);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  const firstKey = post.mock.calls[0][2]?.headers;
  expect(firstKey).toEqual({ "Idempotency-Key": expect.any(String) });
  await act(async () => { request.reject(new Error("Lost response")); expect(await pending).toBe(false); });
  post.mockResolvedValue({ id: "sent" });
  await result.current.message(adminUser.userId, { subject: "Hello", body: "Private" });
  expect(post.mock.calls[1][2]?.headers).toEqual(firstKey);
  await result.current.message(adminUser.userId, { subject: "Hello", body: "Private" });
  expect(post.mock.calls[2][2]?.headers).not.toEqual(firstKey);
});
it.each(actions)("%s retains canonical data while pending and on failure", async (action) => {
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValue(request.promise);
  const { result } = renderHook(useAdminUserActions, { wrapper });
  const pending = invoke(result.current, action);
  expect(client.getQueryData(listKey)).toEqual([adminUser]); expect(client.getQueryData(detailKey)).toEqual(detail);
  expect(useToastStore.getState().toasts).toEqual([]); await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  request.reject(new Error("Backend refused")); await expect(pending).resolves.toBe(false);
  expect(client.getQueryData(listKey)).toEqual([adminUser]); expect(client.getQueryData(detailKey)).toEqual(detail);
  expect(client.getQueryData(settingsKey)).toEqual(adminUserSettings);
  expect(client.getQueryState(listKey)?.isInvalidated).toBe(false);
  expect(useToastStore.getState().toasts).toEqual([expect.objectContaining({ kind: "error" })]);
});
it.each(actions)("%s installs only backend-confirmed fields and invalidates history/settings/KPIs", async (action) => {
  const confirmed = { ...adminUser, brandName: "Backend name", verificationStatus: "approved", suspendedAt: null, avatarUrl: null };
  const otherKey = adminUserKeys.list("other-admin"); client.setQueryData(otherKey, [adminUser]); client.setQueryData(["admin", "kpis"], {});
  vi.spyOn(api, "post").mockResolvedValue(confirmed);
  const { result } = renderHook(useAdminUserActions, { wrapper });
  await expect(invoke(result.current, action)).resolves.toBe(true);
  expect(client.getQueryData(listKey)).toEqual([confirmed]);
  expect(client.getQueryData(detailKey)).toEqual({ ...confirmed, decisionLog: [] });
  for (const key of [detailKey, settingsKey, ["admin", "kpis"]]) expect(client.getQueryState(key)?.isInvalidated).toBe(true);
  expect(client.getQueryState(otherKey)?.isInvalidated).toBe(false); expect(client.getQueryData(otherKey)).toEqual([adminUser]);
});
it.each(["logout", "switch", "same-user-login"])("discards late success and failure after %s", async (transition) => {
  const success = deferred<unknown>(), failure = deferred<unknown>();
  vi.spyOn(api, "post").mockReturnValueOnce(success.promise).mockReturnValueOnce(failure.promise);
  const { result } = renderHook(useAdminUserActions, { wrapper });
  const first = result.current.approve(adminUser.userId), second = result.current.reject("another-user", "Fix", 7);
  await waitFor(() => expect(api.post).toHaveBeenCalledTimes(2));
  act(() => {
    useAuthStore.getState().logout(); client.clear();
    if (transition !== "logout") useAuthStore.getState().setUser(transition === "switch" ? { ...admin, id: "admin-b" } : admin);
  });
  success.resolve({ ...adminUser, verificationStatus: "approved" }); failure.reject(new Error("Late failure"));
  await expect(first).resolves.toBe(false); await expect(second).resolves.toBe(false);
  expect(client.getQueryCache().getAll()).toHaveLength(0); expect(useToastStore.getState().toasts).toEqual([]);
});
it("guards non-admin actions and self-suspension", async () => {
  const post = vi.spyOn(api, "post"); const { result } = renderHook(useAdminUserActions, { wrapper });
  await expect(result.current.suspend(admin.id, "Reason")).resolves.toBe(false);
  expect(useToastStore.getState().toasts[0].message).toContain("own admin account");
  act(() => useAuthStore.getState().setUser({ ...admin, role: "RUNNER" }));
  await expect(result.current.approve(adminUser.userId)).resolves.toBe(false); expect(post).not.toHaveBeenCalled();
});
it("shares directory requests, reads every page, and updates the pending count reactively", async () => {
  client.clear();
  const second = { ...adminUser, userId: "photographer-2", handle: "second" };
  const get = vi.spyOn(api, "get").mockResolvedValueOnce({ items: [adminUser], total: 2 }).mockResolvedValueOnce({ items: [second], total: 2 });
  const { result } = renderHook(() => ({ one: useAdminUsers(), two: useAdminUsers(), count: usePendingVerificationsCount() }), { wrapper });
  await waitFor(() => expect(result.current.count).toBe(2)); expect(get).toHaveBeenCalledTimes(2);
  expect(get.mock.calls[1][0]).toContain("offset=1"); expect(result.current.one.data).toBe(result.current.two.data);
  act(() => client.setQueryData(listKey, [{ ...adminUser, verificationStatus: "approved" }, second]));
  await waitFor(() => expect(result.current.count).toBe(1));
});
it("refetches server-issued decisions and settings after approval", async () => {
  const log = [{ userId: adminUser.userId, decision: "approved", reason: null, decidedAt: "2026-01-01T00:00:00Z" }];
  const confirmed = { ...adminUser, verificationStatus: "approved" };
  vi.spyOn(api, "post").mockResolvedValue(confirmed);
  vi.spyOn(api, "get").mockImplementation(async (path) => path.endsWith("/settings") ? { ...adminUserSettings, bio: "Backend bio" } : { ...confirmed, decisionLog: log });
  const { result } = renderHook(() => ({ detail: useAdminUserDetail(adminUser.userId), settings: useAdminPhotographerSettings(adminUser.userId), actions: useAdminUserActions() }), { wrapper });
  await act(async () => { await result.current.actions.approve(adminUser.userId); });
  await waitFor(() => expect(result.current.detail.data?.decisionLog).toEqual(log));
  expect(result.current.settings.data?.bio).toBe("Backend bio");
});
it("clearing an old session prevents late list/detail/settings reads from repopulating it", async () => {
  client.clear(); const request = deferred<unknown>(); const get = vi.spyOn(api, "get").mockReturnValue(request.promise);
  const hook = renderHook(() => ({ users: useAdminUsers(), detail: useAdminUserDetail(adminUser.userId), settings: useAdminPhotographerSettings(adminUser.userId) }), { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(3));
  hook.unmount(); act(() => { useAuthStore.getState().logout(); client.clear(); });
  await act(async () => request.resolve({ ...detail, items: [adminUser], total: 1 }));
  expect(client.getQueryCache().getAll()).toHaveLength(0);
});
it("cancels an older directory read before installing a confirmed mutation", async () => {
  const old = deferred<unknown>();
  const read = client.fetchQuery({ queryKey: listKey, queryFn: () => old.promise }).catch(() => undefined);
  const confirmed = { ...adminUser, verificationStatus: "approved" };
  vi.spyOn(api, "post").mockResolvedValue(confirmed);
  const { result } = renderHook(useAdminUserActions, { wrapper });
  await result.current.approve(adminUser.userId);
  old.resolve([adminUser]); await read;
  expect(client.getQueryData(listKey)).toEqual([confirmed]);
});
it("does not fetch private user data for a non-admin session", async () => {
  useAuthStore.getState().setUser({ ...admin, role: "RUNNER" });
  const get = vi.spyOn(api, "get");
  const { result } = renderHook(() => ({ users: useAdminUsers(), detail: useAdminUserDetail(adminUser.userId), settings: useAdminPhotographerSettings(adminUser.userId) }), { wrapper });
  expect(result.current.users.fetchStatus).toBe("idle"); expect(result.current.detail.fetchStatus).toBe("idle"); expect(result.current.settings.fetchStatus).toBe("idle");
  expect(get).not.toHaveBeenCalled();
});

it.each([false, true])("waits for a photographer deep link before returning not found (stale empty cache: %s)", async (stale) => {
  client.clear();
  if (stale) client.setQueryData(listKey, [], { updatedAt: 1 });
  const request = deferred<unknown>();
  vi.spyOn(api, "get").mockImplementation(async (path) => path.startsWith("/admin/users?") ? request.promise : path.endsWith("/settings") ? adminUserSettings : detail);
  const { result } = renderHook(() => useAdminPhotographerView(adminUser.handle!), { wrapper });
  expect(result.current.loading).toBe(true); expect(result.current.view).toBeNull();
  await act(async () => request.resolve({ items: [adminUser], total: 1 }));
  await waitFor(() => expect(result.current.view?.row.userId).toBe(adminUser.userId));
  expect(result.current.error).toBeNull();
});
const firstMessage = { subject: "Hello", body: "Private" }, otherMessage = { subject: "Hello", body: "Corrected" };
async function loseMessageReply() {
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Response lost"));
  const { result } = renderHook(useAdminUserActions, { wrapper });
  await expect(result.current.message(adminUser.userId, firstMessage)).resolves.toBe(false);
  post.mockResolvedValue({ id: "sent" }); useToastStore.getState().clear();
  return { post, actions: result.current, reference: (post.mock.calls[0][2]?.headers as Record<string, string>)["Idempotency-Key"] };
}
it.each([
  ["was delivered", () => ({ state: "SUCCEEDED", messageId: "message-1" }), /earlier message.*delivered/i],
  ["was never recorded", () => { throw new ApiError([{ code: "ADMIN_COMMAND_NOT_FOUND", message: "Admin request not found" }], 404); }, null],
])("sends a different message under a new key once the earlier one %s", async (_name, answer, notice) => {
  vi.useFakeTimers({ toFake: ["Date"] });
  const { post, actions, reference } = await loseMessageReply();
  const get = vi.spyOn(api, "get").mockImplementation(async () => answer());
  if (!notice) {
    // "No record" is trusted only once the earlier request can no longer be running.
    await expect(actions.message(adminUser.userId, otherMessage)).resolves.toBe(false);
    expect(post).toHaveBeenCalledTimes(1);
    expect(useToastStore.getState().toasts.at(-1)?.message).toMatch(/about 5 minutes/);
    expect(actions.messageRecoveryNotice(adminUser.userId)).toContain(reference);
    useToastStore.getState().clear(); vi.setSystemTime(Date.now() + 5 * 60_000);
  }
  await expect(actions.message(adminUser.userId, otherMessage)).resolves.toBe(true);
  expect(get).toHaveBeenCalledWith(`/admin/commands/${reference}`);
  expect(post.mock.calls[1][2]?.headers).not.toEqual(post.mock.calls[0][2]?.headers);
  expect(useToastStore.getState().toasts.map((toast) => toast.message).filter((message) => /earlier/i.test(message)))
    .toEqual(notice ? [expect.stringMatching(notice)] : []);
  expect(actions.messageRecoveryNotice(adminUser.userId)).toBeNull();
});
it.each([
  ["PENDING", () => ({ state: "PENDING" })], ["UNKNOWN", () => ({ state: "UNKNOWN" })],
  ["a lookup failure", () => { throw new Error("Offline"); }],
])("keeps a different message blocked on %s and shows the reference", async (_name, answer) => {
  const { post, actions, reference } = await loseMessageReply();
  vi.spyOn(api, "get").mockImplementation(async () => answer());
  await expect(actions.message(adminUser.userId, otherMessage)).resolves.toBe(false);
  expect(post).toHaveBeenCalledTimes(1);
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain(reference);
  expect(actions.messageRecoveryNotice(adminUser.userId)).toContain(reference);
});
it.each([400, 403, 404, 422])("releases a message the server refused outright with %s so a corrected one is a new request", async (status) => {
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new ApiError([{ code: "REFUSED", message: "Refused" }], status));
  const get = vi.spyOn(api, "get");
  const { result } = renderHook(useAdminUserActions, { wrapper });
  await expect(result.current.message(adminUser.userId, firstMessage)).resolves.toBe(false);
  expect(useToastStore.getState().toasts.at(-1)?.message).toBe("Refused");
  expect(result.current.messageRecoveryNotice(adminUser.userId)).toBeNull();
  post.mockResolvedValue({ id: "sent" });
  await expect(result.current.message(adminUser.userId, otherMessage)).resolves.toBe(true);
  expect(get).not.toHaveBeenCalled();
  expect(post.mock.calls[1][2]?.headers).not.toEqual(post.mock.calls[0][2]?.headers);
});
it("keeps the key when a retry of an uncertain message is refused, because the first attempt may have been delivered", async () => {
  const { post, actions, reference } = await loseMessageReply();
  post.mockRejectedValueOnce(new ApiError([{ code: "USER_NOT_FOUND", message: "Gone" }], 404));
  await expect(actions.message(adminUser.userId, firstMessage)).resolves.toBe(false);
  expect(actions.messageRecoveryNotice(adminUser.userId)).toContain(reference);
});
