import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { inboxKeys, useMyPhotographerMessages, useMyRunnerMessages } from "./use-inbox-messages";

const message = { id: "message-1", kind: "admin_message", title: "Hello", body: "Update", createdAt: "2026-09-10T00:00:00Z", readAt: null, sourceDecisionId: null, orderId: null };
function deferred<T>() {
  let resolve!: (value: T) => void; let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
describe.each(["PHOTOGRAPHER", "RUNNER"] as const)("%s inbox", (role) => {
  let client: QueryClient;
  const user = { id: "owner-a", role, name: "Owner", email: "owner@example.test", createdAt: "2026-09-10T00:00:00Z" };
  const useMessages = role === "PHOTOGRAPHER" ? useMyPhotographerMessages : useMyRunnerMessages;
  function wrapper({ children }: { children: ReactNode }) { return <QueryClientProvider client={client}>{children}</QueryClientProvider>; }
  const key = () => inboxKeys.messages(role, user.id, useAuthStore.getState().sessionVersion);
  beforeEach(() => {
    useAuthStore.getState().setUser(user); useToastStore.getState().clear();
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  });
  afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

  it("shares requests/pages across bell and modal and paginates beyond 200", async () => {
    const rows = Array.from({ length: 205 }, (_, i) => ({ ...message, id: `message-${i}` }));
    const get = vi.spyOn(api, "getWithTotal").mockImplementation(async (path) => {
      const offset = Number(new URL(path, "https://example.test").searchParams.get("offset"));
      return { data: rows.slice(offset, offset + 100), total: rows.length };
    });
    const { result } = renderHook(() => ({ bell: useMessages(), modal: useMessages() }), { wrapper });
    await waitFor(() => expect(result.current.bell.messages).toHaveLength(100));
    expect(get).toHaveBeenCalledTimes(1);
    await act(async () => { await result.current.modal.loadMore(); });
    await waitFor(() => expect(result.current.bell.messages).toHaveLength(200));
    await act(async () => { await result.current.modal.loadMore(); });
    await waitFor(() => expect(result.current.bell.messages).toHaveLength(205)); expect(result.current.modal.hasMore).toBe(false);
    expect(get.mock.calls.map(([path]) => path)).toEqual([0, 100, 200].map((offset) => `/me/${role.toLowerCase()}/messages?limit=100&offset=${offset}`));
    expect(get.mock.calls[0][1]).toBeInstanceOf(AbortSignal);
  });
  it("retains pages on load-more failure, blocks duplicates, and retries the same offset", async () => {
    const rows = Array.from({ length: 100 }, (_, i) => ({ ...message, id: `message-${i}` }));
    const next = deferred<{ data: unknown[]; total: number }>();
    const get = vi.spyOn(api, "getWithTotal").mockResolvedValueOnce({ data: rows, total: 101 }).mockReturnValueOnce(next.promise).mockResolvedValue({ data: [{ ...message, id: "older" }], total: 101 });
    const { result } = renderHook(() => useMessages(), { wrapper });
    await waitFor(() => expect(result.current.messages).toHaveLength(100));
    let request: unknown;
    act(() => { request = result.current.loadMore(); result.current.loadMore(); });
    expect(get).toHaveBeenCalledTimes(2);
    await act(async () => { next.reject(new Error("Offline")); await request; });
    expect(result.current.messages).toHaveLength(100); await waitFor(() => expect(result.current.error).toBeTruthy()); expect(result.current.hasMore).toBe(true);
    await act(async () => { await result.current.loadMore(); });
    await waitFor(() => expect(result.current.messages).toHaveLength(101)); expect(result.current.error).toBeNull();
    expect(get.mock.calls[1][0]).toBe(get.mock.calls[2][0]);
  });
  it("continues a full page without the total header, dedupes overlapping rows, and stops on an empty page", async () => {
    const rows = Array.from({ length: 100 }, (_, i) => ({ ...message, id: `message-${i}` }));
    vi.spyOn(api, "getWithTotal").mockResolvedValueOnce({ data: rows, total: null }).mockResolvedValueOnce({ data: rows, total: null }).mockResolvedValue({ data: [], total: null });
    const { result } = renderHook(() => useMessages(), { wrapper });
    await waitFor(() => expect(result.current.messages).toHaveLength(100));
    await act(async () => { await result.current.loadMore(); });
    expect(result.current.messages).toHaveLength(100);
    await act(async () => { await result.current.loadMore(); });
    await waitFor(() => expect(result.current.hasMore).toBe(false)); expect(result.current.total).toBeNull();
  });
  it.each(["markRead", "markAllRead", "remove"] as const)("%s retains current data on failure and supports confirmed retry", async (action) => {
    let serverRows = [message];
    vi.spyOn(api, "getWithTotal").mockImplementation(async () => ({ data: serverRows, total: serverRows.length }));
    const request = deferred<unknown>();
    const transport = vi.spyOn(api, action === "remove" ? "delete" : "fetch").mockReturnValueOnce(request.promise);
    const { result } = renderHook(() => ({ bell: useMessages(), modal: useMessages() }), { wrapper });
    await waitFor(() => expect(result.current.bell.messages).toHaveLength(1));
    let pending!: Promise<boolean>;
    act(() => { pending = result.current.modal[action](message.id); });
    await waitFor(() => expect(result.current.bell.pending).toBe(true));
    expect(result.current.bell.messages[0].readAt).toBeNull();
    await expect(result.current.bell[action](message.id)).resolves.toBe(false);
    expect(transport).toHaveBeenCalledTimes(1);
    // A refetch can add a row while a mutation is pending. Failure must not
    // roll that newer data back to a snapshot from before the request.
    serverRows = [{ ...message, id: "new-message" }, message];
    await act(async () => { await result.current.bell.refetch(); });
    await act(async () => { request.reject(new Error("Refused")); expect(await pending).toBe(false); });
    await waitFor(() => expect(result.current.bell.messages).toHaveLength(2)); expect(result.current.bell.total).toBe(2);
    expect(result.current.bell.messages.every((m) => m.readAt === null)).toBe(true);
    expect(useToastStore.getState().toasts).toEqual([expect.objectContaining({ kind: "error" })]);
    const confirmed = { ...message, readAt: "2026-09-10T10:11:12Z" };
    transport.mockImplementationOnce(async () => {
      serverRows = action === "remove" ? [serverRows[0]] : serverRows.map((m) => action === "markAllRead" || m.id === message.id ? { ...m, readAt: confirmed.readAt } : m) as typeof serverRows;
      return action === "remove" ? { removed: true } : action === "markAllRead" ? { markedRead: 2 } : confirmed;
    });
    await act(async () => { expect(await result.current.modal[action](message.id)).toBe(true); });
    await waitFor(() => expect(result.current.bell.pending).toBe(false));
    if (action === "remove") { expect(result.current.bell.messages).toHaveLength(1); expect(result.current.bell.total).toBe(1); }
    else expect(result.current.bell.messages.find((m) => m.id === message.id)?.readAt).toBe(confirmed.readAt);
    expect(result.current.bell.messages).toEqual(result.current.modal.messages);
  });
  it.each(["logout", "switch", "same-user-login"])("discards old reads and mutation outcomes after %s", async (transition) => {
    const request = deferred<unknown>();
    vi.spyOn(api, "fetch").mockReturnValue(request.promise);
    const read = deferred<{ data: unknown[]; total: number }>();
    vi.spyOn(api, "getWithTotal").mockReturnValue(read.promise);
    const { result, unmount } = renderHook(() => useMessages(), { wrapper });
    const oldKey = key();
    let pending!: Promise<boolean>;
    act(() => { pending = result.current.markRead(message.id); });
    await waitFor(() => expect(result.current.pending).toBe(true));
    unmount();
    act(() => {
      useAuthStore.getState().logout(); client.clear();
      if (transition !== "logout") useAuthStore.getState().setUser(transition === "switch" ? { ...user, id: "owner-b" } : user);
    });
    read.resolve({ data: [message], total: 1 }); request.resolve({ ...message, readAt: "2026-09-10T00:00:00Z" });
    await expect(pending).resolves.toBe(false);
    expect(client.getQueryData(oldKey)).toBeUndefined(); expect(useToastStore.getState().toasts).toEqual([]);
  });
  it("suppresses stale errors, and a callback retained from an old session cannot act on a new one", async () => {
    const request = deferred<unknown>(); const patch = vi.spyOn(api, "fetch").mockReturnValue(request.promise);
    const { result } = renderHook(() => useMessages(false), { wrapper });
    const oldAction = result.current.markRead;
    let pending!: Promise<boolean>; act(() => { pending = oldAction(message.id); });
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1));
    act(() => { useAuthStore.getState().logout(); client.clear(); useAuthStore.getState().setUser(user); });
    await act(async () => { request.reject(new Error("Late failure")); expect(await pending).toBe(false); });
    await expect(oldAction(message.id)).resolves.toBe(false);
    expect(patch).toHaveBeenCalledTimes(1); expect(useToastStore.getState().toasts).toEqual([]);
  });
  it("gates reads and actions by true role and authentication", async () => {
    useAuthStore.getState().setUser({ ...user, role: "ADMIN" });
    const get = vi.spyOn(api, "getWithTotal"); const patch = vi.spyOn(api, "fetch");
    const { result } = renderHook(() => useMessages(), { wrapper });
    await expect(result.current.markRead(message.id)).resolves.toBe(false);
    expect(get).not.toHaveBeenCalled(); expect(patch).not.toHaveBeenCalled();
    act(() => useAuthStore.getState().logout());
    await expect(result.current.remove(message.id)).resolves.toBe(false); expect(get).not.toHaveBeenCalled();
  });
  it("cancels an older read before reconciling a confirmed mutation", async () => {
    const stale = deferred<{ data: unknown[]; total: number }>();
    const confirmed = { ...message, readAt: "2026-09-10T10:11:12Z" };
    const get = vi.spyOn(api, "getWithTotal").mockResolvedValueOnce({ data: [message], total: 1 }).mockReturnValueOnce(stale.promise).mockResolvedValue({ data: [confirmed], total: 1 });
    vi.spyOn(api, "fetch").mockResolvedValue(confirmed);
    const { result } = renderHook(() => useMessages(), { wrapper });
    await waitFor(() => expect(result.current.messages).toHaveLength(1));
    act(() => { void result.current.refetch(); });
    await act(async () => { expect(await result.current.markRead(message.id)).toBe(true); });
    await waitFor(() => expect(result.current.messages[0].readAt).toBe(confirmed.readAt));
    expect(get.mock.calls[1][1]?.aborted).toBe(true);
    await act(async () => stale.resolve({ data: [message], total: 1 }));
    expect(result.current.messages[0].readAt).toBe(confirmed.readAt);
  });
});
