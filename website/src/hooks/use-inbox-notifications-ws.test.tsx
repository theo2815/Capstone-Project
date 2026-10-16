import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useInboxNotificationsWs } from "./use-inbox-notifications-ws";
import { inboxKeys, useMyPhotographerMessages, useMyRunnerMessages } from "./use-inbox-messages";
import { removalKeys } from "./use-photo-removal";

vi.mock("@/lib/auth", () => ({ getAccessToken: () => "fixture-token" }));
class Socket {
  onopen = () => {};
  onclose = () => {};
  onmessage: (event: { data: string }) => void = () => {};
  close = vi.fn();
}
const message = { id: "message-1", kind: "admin_message", title: "Hello", body: "Update", createdAt: "2026-09-10T00:00:00Z", readAt: null, sourceDecisionId: null, orderId: null };
describe.each(["PHOTOGRAPHER", "RUNNER"] as const)("%s realtime inbox", (role) => {
  let client: QueryClient;
  let sockets: Socket[];
  const user = { id: "owner-a", role, name: "Owner", email: "owner@example.test", createdAt: "2026-09-10T00:00:00Z" };
  const useMessages = role === "PHOTOGRAPHER" ? useMyPhotographerMessages : useMyRunnerMessages;
  function wrapper({ children }: { children: ReactNode }) { return <QueryClientProvider client={client}>{children}</QueryClientProvider>; }
  beforeEach(() => {
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    useAuthStore.getState().setUser(user); sockets = [];
    vi.stubGlobal("WebSocket", vi.fn(function () { const socket = new Socket(); sockets.push(socket); return socket; }));
  });
  afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });
  if (role === "PHOTOGRAPHER") it("refreshes scoped removal cases on reconnect and durable removal notices", async () => {
    const key = removalKeys.photographer([user.id, role, useAuthStore.getState().sessionVersion]);
    client.setQueryData(key, []);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    renderHook(() => useInboxNotificationsWs(role, true), { wrapper });
    await act(async () => sockets[0].onopen());
    expect(invalidate).toHaveBeenCalledWith({ queryKey: key });
    invalidate.mockClear();
    await act(async () => sockets[0].onmessage({ data: JSON.stringify({ type: "message.created", message: { ...message, photoRemovalRequestId: "case" } }) }));
    expect(invalidate).toHaveBeenCalledWith({ queryKey: key });
  });
  it("refreshes authoritative pages on connect and duplicate pushes, ignoring malformed frames", async () => {
    let rows: typeof message[] = [];
    const get = vi.spyOn(api, "getWithTotal").mockImplementation(async () => ({ data: rows, total: rows.length }));
    const { result } = renderHook(() => ({ connected: useInboxNotificationsWs(role, true), inbox: useMessages() }), { wrapper });
    await waitFor(() => expect(result.current.inbox.total).toBe(0));
    rows = [message];
    act(() => sockets[0].onopen());
    await waitFor(() => expect(result.current.inbox.messages).toEqual([message]));
    expect(result.current.connected).toBe(true);
    const calls = get.mock.calls.length;
    act(() => { sockets[0].onmessage({ data: "bad JSON" }); sockets[0].onmessage({ data: JSON.stringify({ type: "other" }) }); });
    expect(get).toHaveBeenCalledTimes(calls);
    const frame = { data: JSON.stringify({ type: "message.created", message: { ...message, title: "Untrusted push title" } }) };
    await act(async () => { sockets[0].onmessage(frame); sockets[0].onmessage(frame); });
    await waitFor(() => expect(get.mock.calls.length).toBeGreaterThan(calls));
    expect(result.current.inbox.messages).toEqual([message]); expect(result.current.inbox.total).toBe(1);
  });
  it("cancels an initial stale read when a push arrives before it resolves", async () => {
    let resolve!: (value: { data: unknown[]; total: number }) => void;
    const stale = new Promise<{ data: unknown[]; total: number }>((yes) => { resolve = yes; });
    const get = vi.spyOn(api, "getWithTotal").mockReturnValueOnce(stale).mockResolvedValue({ data: [message], total: 1 });
    const { result } = renderHook(() => { useInboxNotificationsWs(role, true); return useMessages(); }, { wrapper });
    act(() => sockets[0].onmessage({ data: JSON.stringify({ type: "message.created", message }) }));
    await waitFor(() => expect(result.current.messages).toHaveLength(1));
    expect(get.mock.calls[0][1]?.aborted).toBe(true);
    await act(async () => resolve({ data: [], total: 0 }));
    expect(result.current.messages).toHaveLength(1);
  });
  it.each(["switch", "same-user-login", "logout"])("ignores queued frames and reconnects from old sockets after %s", async (transition) => {
    vi.spyOn(api, "getWithTotal").mockResolvedValue({ data: [], total: 0 });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const { unmount } = renderHook(() => useInboxNotificationsWs(role, true), { wrapper });
    const old = sockets[0];
    act(() => {
      useAuthStore.getState().logout(); client.clear();
      if (transition !== "logout") useAuthStore.getState().setUser(transition === "switch" ? { ...user, id: "owner-b" } : user);
    });
    invalidate.mockClear();
    await act(async () => { old.onopen(); old.onmessage({ data: JSON.stringify({ type: "message.created", message }) }); old.onclose(); });
    expect(invalidate).not.toHaveBeenCalled(); expect(old.close).toHaveBeenCalled();
    expect(sockets).toHaveLength(transition === "logout" ? 1 : 2);
    if (transition !== "logout") {
      await act(async () => sockets[1].onopen());
      expect(invalidate).toHaveBeenCalledWith({ queryKey: inboxKeys.messages(role, transition === "switch" ? "owner-b" : user.id, useAuthStore.getState().sessionVersion), exact: true });
    }
    unmount(); invalidate.mockClear();
    await act(async () => sockets.at(-1)!.onmessage({ data: JSON.stringify({ type: "message.created", message }) }));
    expect(invalidate).not.toHaveBeenCalled();
  });
  it("polls every 30 seconds while disconnected, every 5 minutes while connected, and reconnects", async () => {
    vi.useFakeTimers();
    const get = vi.spyOn(api, "getWithTotal").mockResolvedValue({ data: [], total: 0 });
    const { result, unmount } = renderHook(() => {
      const connected = useInboxNotificationsWs(role, true);
      return { connected, inbox: useMessages(true, connected ? 300_000 : 30_000) };
    }, { wrapper });
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(get).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(get).toHaveBeenCalledTimes(2);
    await act(async () => sockets[0].onopen());
    const calls = get.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(get).toHaveBeenCalledTimes(calls);
    await act(async () => { await vi.advanceTimersByTimeAsync(270_001); });
    expect(get).toHaveBeenCalledTimes(calls + 1);
    act(() => sockets[0].onclose()); expect(result.current.connected).toBe(false);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(sockets).toHaveLength(2);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(sockets).toHaveLength(2);
  });
});
