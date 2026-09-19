import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAdminPhotographerView } from "@/lib/admin-photographer-view";
import { api } from "@/lib/api";
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
  useAuthStore.getState().setUser(admin); useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(listKey, [adminUser]); client.setQueryData(detailKey, detail); client.setQueryData(settingsKey, adminUserSettings);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
const actions = ["approve", "reject", "suspend", "unsuspend", "resetVerification"] as const;
function invoke(actionsHook: ReturnType<typeof useAdminUserActions>, action: typeof actions[number]) {
  return actionsHook[action](adminUser.userId, "Please fix the watermark");
}
it.each(actions)("%s retains canonical data while pending and on failure", async (action) => {
  const request = deferred<unknown>(); const post = vi.spyOn(api, "post").mockReturnValue(request.promise);
  const { result } = renderHook(useAdminUserActions, { wrapper });
  const pending = invoke(result.current, action);
  expect(client.getQueryData(listKey)).toEqual([adminUser]); expect(client.getQueryData(detailKey)).toEqual(detail);
  expect(useToastStore.getState().toasts).toEqual([]); expect(post).toHaveBeenCalledTimes(1);
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
  const first = result.current.approve(adminUser.userId), second = result.current.reject(adminUser.userId, "Fix");
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
