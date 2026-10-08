import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { webcrypto } from "node:crypto";
import { clearAdminActionIntents } from "@/lib/admin-action-intents";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { admin, adminEvent } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { adminEventKeys, useAdminEvents, useAdminEventActions } from "./use-admin-events";
import { usePublicEvents } from "./use-public-events";

let client: QueryClient;
const key = adminEventKeys.list(admin.id);
const publicKey = ["public", "events", {}];
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto); clearAdminActionIntents();
  useAuthStore.getState().setUser(admin); useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(key, [adminEvent]); client.setQueryData(publicKey, [adminEvent]);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); clearAdminActionIntents(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });
const createArgs = { title: "New race", date: adminEvent.date, location: "Cebu", pricePerPhoto: 80 };
const actions = ["createEvent", "editEvent", "deleteEvent", "approve", "reject"] as const;
it("claims an event before hashing, blocks another view and reuses the update key after timeout", async () => {
  const request = deferred<unknown>(); const patch = { title: "Race", expectedRevision: 4, cover: new File(["cover"], "cover.jpg") };
  const update = vi.spyOn(api, "fetch").mockReturnValueOnce(request.promise); const remove = vi.spyOn(api, "delete");
  const { result } = renderHook(() => [useAdminEventActions(), useAdminEventActions()], { wrapper });
  let first!: Promise<boolean>;
  act(() => { first = result.current[0].editEvent(adminEvent.id, patch); });
  const rejected = expect(first).rejects.toThrow();
  await expect(result.current[1].deleteEvent(adminEvent.id)).resolves.toBe(false);
  expect(remove).not.toHaveBeenCalled();
  await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
  const firstKey = update.mock.calls[0][1]?.headers;
  expect(firstKey).toEqual({ "Idempotency-Key": expect.any(String) });
  expect((update.mock.calls[0][1]?.body as FormData).get("expectedRevision")).toBe("4");
  request.reject(new ApiError([{ code: "TIMEOUT", message: "Response lost" }])); await rejected;
  update.mockResolvedValue(adminEvent);
  await result.current[0].editEvent(adminEvent.id, patch);
  expect(update.mock.calls[1][1]?.headers).toEqual(firstKey);
});
it("preserves create identity after UNKNOWN and starts a new intent only after confirmation", async () => {
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new ApiError([{ code: "ADMIN_COMMAND_RECOVERY_REQUIRED", message: "Unknown" }], 409));
  const { result } = renderHook(useAdminEventActions, { wrapper });
  await expect(result.current.createEvent(createArgs)).rejects.toThrow();
  const firstKey = post.mock.calls[0][2]?.headers;
  expect(firstKey).toEqual({ "Idempotency-Key": expect.any(String) });
  post.mockResolvedValue(adminEvent);
  await result.current.createEvent(createArgs);
  expect(post.mock.calls[1][2]?.headers).toEqual(firstKey);
  await result.current.createEvent(createArgs);
  expect(post.mock.calls[2][2]?.headers).not.toEqual(firstKey);
});
it("sends the reviewed event revision and does not fail an accepted command on refresh error", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue(adminEvent);
  vi.spyOn(client, "invalidateQueries").mockRejectedValue(new Error("Refresh failed"));
  const { result } = renderHook(useAdminEventActions, { wrapper });
  await expect(result.current.approve(adminEvent.id, 9)).resolves.toBe(true);
  expect(post).toHaveBeenCalledWith(`/admin/events/${adminEvent.id}/approve`, { expectedRevision: 9 });
  expect(client.getQueryData(publicKey)).toEqual([adminEvent]);
});
it("allows correcting a cover rejected before execution but retains keys for uncertain responses", async () => {
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new ApiError([{ code: "UNSUPPORTED_MEDIA_TYPE", field: "cover", message: "Cannot decode cover" }], 400));
  const { result } = renderHook(useAdminEventActions, { wrapper });
  await expect(result.current.createEvent({ ...createArgs, cover: new File(["bad"], "bad.jpg") })).rejects.toThrow();
  post.mockResolvedValue(adminEvent);
  await expect(result.current.createEvent({ ...createArgs, cover: new File(["valid"], "good.jpg") })).resolves.toBe(true);
  expect(post).toHaveBeenCalledTimes(2);
  expect(post.mock.calls[1][2]?.headers).not.toEqual(post.mock.calls[0][2]?.headers);
});
function invoke(apiActions: ReturnType<typeof useAdminEventActions>, name: typeof actions[number]) {
  return name === "createEvent" ? apiActions.createEvent(createArgs)
    : name === "editEvent" ? apiActions.editEvent(adminEvent.id, { title: "Changed" })
    : name === "reject" ? apiActions.reject(adminEvent.id, "Please fix pricing")
    : apiActions[name](adminEvent.id);
}
function mockRequest(action: typeof actions[number], promise: Promise<unknown>) {
  return vi.spyOn(api, action === "deleteEvent" ? "delete" : action === "editEvent" ? "fetch" : "post")
    .mockReturnValue(promise);
}
it.each(actions)("%s retains canonical data while pending and on failure", async (action) => {
  const request = deferred<unknown>(); const endpoint = mockRequest(action, request.promise);
  const { result } = renderHook(useAdminEventActions, { wrapper });
  const pending = invoke(result.current, action);
  const rejected = expect(pending).rejects.toThrow("Backend refused");
  expect(client.getQueryData(key)).toEqual([adminEvent]);
  expect(useToastStore.getState().toasts).toHaveLength(0);
  await waitFor(() => expect(endpoint).toHaveBeenCalledTimes(1));
  request.reject(new Error("Backend refused")); await rejected;
  expect(client.getQueryData(key)).toEqual([adminEvent]);
  expect(client.getQueryState(key)?.isInvalidated).toBe(false);
  expect(client.getQueryData(publicKey)).toEqual([adminEvent]);
  expect(useToastStore.getState().toasts).toEqual([expect.objectContaining({ kind: "error" })]);
});
it.each(["editEvent", "approve", "reject"] as const)("%s replaces complete backend rows and reconciles filters", async (action) => {
  const reviewKey = adminEventKeys.list(admin.id, { review: "queue" });
  const liveKey = adminEventKeys.list(admin.id, { state: "live" });
  const otherKey = adminEventKeys.list("another-admin");
  [reviewKey, liveKey, otherKey].forEach((k) => client.setQueryData(k, [adminEvent]));
  const confirmed = { ...adminEvent, name: "Backend title", date: "2026-12-01", state: "upcoming", bannerUrl: undefined };
  client.setQueryData(key, [{ ...adminEvent, bannerUrl: "old-cover" }]);
  mockRequest(action, Promise.resolve(confirmed));
  const { result } = renderHook(useAdminEventActions, { wrapper });
  await expect(invoke(result.current, action)).resolves.toBe(true);
  expect(client.getQueryData(key)).toEqual([confirmed]);
  expect(client.getQueryData(reviewKey)).toEqual([]); expect(client.getQueryData(liveKey)).toEqual([]);
  expect(client.getQueryData(otherKey)).toEqual([adminEvent]);
  expect(client.getQueryState(otherKey)?.isInvalidated).toBe(false);
  expect(client.getQueryData(publicKey)).toEqual([adminEvent]);
  expect(client.getQueryState(publicKey)?.isInvalidated).toBe(true);
});
it("creates once from the backend DTO without inserting private records into public or review lists", async () => {
  const created = { ...adminEvent, id: "created", slug: "backend-slug", visibility: "unlisted" };
  const reviewKey = adminEventKeys.list(admin.id, { review: "queue" });
  client.setQueryData(reviewKey, []);
  const post = vi.spyOn(api, "post").mockResolvedValue(created);
  const { result } = renderHook(useAdminEventActions, { wrapper });
  await result.current.createEvent({ ...createArgs, cover: new File(["cover"], "cover.jpg") });
  await result.current.createEvent(createArgs);
  expect(client.getQueryData(key)).toEqual([created, adminEvent]);
  expect(client.getQueryData(reviewKey)).toEqual([]);
  expect(client.getQueryData(publicKey)).toEqual([adminEvent]);
  const form = post.mock.calls[0][1] as FormData;
  expect(form.get("title")).toBe(createArgs.title); expect(form.get("cover")).toBeInstanceOf(File);
});
it.each([true, false])("deletes only after confirmation, including already-deleted response (%s)", async (removed) => {
  const request = deferred<unknown>(); vi.spyOn(api, "delete").mockReturnValue(request.promise);
  const { result } = renderHook(useAdminEventActions, { wrapper });
  const pending = result.current.deleteEvent(adminEvent.id);
  expect(client.getQueryData(key)).toEqual([adminEvent]); request.resolve({ removed });
  await expect(pending).resolves.toBe(true); expect(client.getQueryData(key)).toEqual([]);
});
it("cancels older admin reads and allows later refetches to supersede an edit", async () => {
  const old = deferred<unknown>();
  const get = vi.spyOn(api, "get").mockReturnValueOnce(old.promise);
  const read = client.fetchQuery({ queryKey: key, queryFn: () => old.promise }).catch(() => undefined);
  const confirmed = { ...adminEvent, name: "Confirmed" };
  vi.spyOn(api, "fetch").mockResolvedValue(confirmed);
  const { result } = renderHook(useAdminEventActions, { wrapper });
  await result.current.editEvent(adminEvent.id, { title: "Draft" });
  old.resolve([adminEvent]); await read;
  expect(client.getQueryData(key)).toEqual([confirmed]);
  const later = { ...confirmed, name: "Later backend value" };
  get.mockReset().mockResolvedValue({ items: [later] });
  const query = renderHook(useAdminEvents, { wrapper });
  await waitFor(() => expect(query.result.current.data).toEqual([later]));
});
it("refetches conflicting review decisions without committing a failed request", async () => {
  vi.spyOn(api, "post").mockRejectedValue(new ApiError([{ code: "CONFLICT", message: "Already reviewed" }], 409));
  const confirmed = { ...adminEvent, reviewStatus: "rejected" };
  vi.spyOn(api, "get").mockResolvedValue({ items: [confirmed] });
  const { result } = renderHook(() => ({ query: useAdminEvents(), actions: useAdminEventActions() }), { wrapper });
  await act(async () => { await expect(result.current.actions.approve(adminEvent.id)).rejects.toThrow(); });
  await waitFor(() => expect(result.current.query.data).toEqual([confirmed]));
});
it.each(["logout", "account-change", "same-user-relogin"])("discards all late mutation outcomes after %s", async (transition) => {
  const requests = actions.map(() => deferred<unknown>());
  vi.spyOn(api, "post").mockImplementation((path) => path.endsWith("/approve") ? requests[3].promise : path.endsWith("/reject") ? requests[4].promise : requests[0].promise);
  vi.spyOn(api, "fetch").mockReturnValue(requests[1].promise); vi.spyOn(api, "delete").mockReturnValue(requests[2].promise);
  const { result } = renderHook(useAdminEventActions, { wrapper });
  const pending = [result.current.createEvent(createArgs), result.current.editEvent("edit", { title: "Changed" }), result.current.deleteEvent("delete"), result.current.approve("approve", 4), result.current.reject("reject", "Fix", 4)];
  await waitFor(() => expect(api.post).toHaveBeenCalledTimes(3));
  expect(api.fetch).toHaveBeenCalledTimes(1); expect(api.delete).toHaveBeenCalledTimes(1);
  useAuthStore.getState().logout(); client.clear();
  if (transition !== "logout") useAuthStore.getState().setUser({ ...admin, id: transition === "account-change" ? "admin-b" : admin.id });
  const newKey = adminEventKeys.list(useAuthStore.getState().user?.id ?? null); client.setQueryData(newKey, []);
  requests.forEach((request, i) => i === 4 ? request.reject(new Error("Late failure")) : request.resolve(i === 2 ? { removed: true } : adminEvent));
  expect(await Promise.all(pending)).toEqual(actions.map(() => false));
  expect(client.getQueryData(newKey)).toEqual([]); expect(client.getQueryState(newKey)?.isInvalidated).toBe(false);
  expect(useToastStore.getState().toasts).toEqual([]);
});
it("shares public SSR data through Query and refetches after admin invalidation", async () => {
  client.removeQueries({ queryKey: publicKey });
  let items = [adminEvent];
  vi.spyOn(api, "get").mockImplementation(async () => ({ items }));
  vi.spyOn(api, "delete").mockImplementation(async () => { items = []; return { removed: true }; });
  const { result } = renderHook(() => ({ events: usePublicEvents({}, [adminEvent]), actions: useAdminEventActions() }), { wrapper });
  await waitFor(() => expect(client.isFetching()).toBe(0));
  await act(async () => { await result.current.actions.deleteEvent(adminEvent.id); });
  await waitFor(() => expect(result.current.events).toEqual([]));
});
it("does not fetch or mutate as a signed-out visitor", async () => {
  useAuthStore.getState().logout(); const get = vi.spyOn(api, "get"); const post = vi.spyOn(api, "post");
  const { result } = renderHook(() => ({ query: useAdminEvents(), actions: useAdminEventActions() }), { wrapper });
  await expect(result.current.actions.createEvent(createArgs)).resolves.toBe(false);
  expect(get).not.toHaveBeenCalled(); expect(post).not.toHaveBeenCalled();
});
it("cancels an older public fetch before reconciling an event deletion", async () => {
  client.removeQueries({ queryKey: publicKey });
  const old = deferred<unknown>();
  vi.spyOn(api, "get").mockReturnValueOnce(old.promise).mockResolvedValue({ items: [] });
  vi.spyOn(api, "delete").mockResolvedValue({ removed: true });
  const { result } = renderHook(() => ({ events: usePublicEvents(), actions: useAdminEventActions() }), { wrapper });
  await act(async () => { await result.current.actions.deleteEvent(adminEvent.id); });
  await act(async () => { old.resolve({ items: [adminEvent] }); });
  await waitFor(() => expect(result.current.events).toEqual([]));
});
it("forwards multipart edit fields and cover removal without optimistic patches", async () => {
  const update = vi.spyOn(api, "fetch").mockResolvedValue(adminEvent);
  const { result } = renderHook(useAdminEventActions, { wrapper });
  await result.current.editEvent(adminEvent.id, { title: "Title", pricePerPhoto: 120, removeCover: true });
  const form = update.mock.calls[0][1]?.body as FormData;
  expect(update.mock.calls[0][0]).toBe(`/admin/events/${adminEvent.id}`);
  expect(form.get("pricePerPhoto")).toBe("120"); expect(form.get("removeCover")).toBe("true");
  expect(form.has("cover")).toBe(false);
});
