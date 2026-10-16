import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import { useAdminNotificationsWs } from "./use-admin-notifications-ws";
import { removalKeys } from "./use-photo-removal";

vi.mock("@/lib/auth", () => ({ getAccessToken: () => "fixture-token" }));


const admin = { id: "admin-a", role: "ADMIN" as const, name: "Admin", email: "admin@example.test", createdAt: "2026-09-10" };
beforeEach(() => useAuthStore.getState().setUser(admin));
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); useAuthStore.getState().logout(); });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

it("refreshes canonical queues on reconnect and filed events, and ignores messages after teardown", () => {
  const socket = { onopen: () => {}, onmessage: vi.fn<(event: { data: string }) => void>(), close: vi.fn() };
  vi.stubGlobal("WebSocket", vi.fn(function () { return socket; }));
  const client = new QueryClient();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { unmount } = renderHook(() => useAdminNotificationsWs(true), { wrapper });
  socket.onopen();
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "users"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "disputes"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "payouts"] });
  invalidate.mockClear();
  socket.onmessage({ data: JSON.stringify({ type: "verification_submitted" }) });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "users"] });
  socket.onmessage({ data: JSON.stringify({ type: "dispute_filed" }) });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "disputes"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "kpis"] });
  socket.onmessage({ data: JSON.stringify({ type: "payout_report_filed" }) });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "payouts", "reports"] });
  // The old socket must stop affecting caches immediately on logout, even
  // before React has run the effect teardown.
  invalidate.mockClear();
  act(() => {
    useAuthStore.getState().logout();
    socket.onmessage({ data: JSON.stringify({ type: "dispute_filed" }) });
  });
  expect(invalidate).not.toHaveBeenCalled();
  unmount();
  invalidate.mockClear();
  socket.onmessage({ data: JSON.stringify({ type: "dispute_filed" }) });
  expect(invalidate).not.toHaveBeenCalled();
  client.clear();
});

it("refreshes current-session removal cases on reconnect and removal changes", async () => {
  const socket = { onopen: () => {}, onmessage: vi.fn<(event: { data: string }) => void>(), close: vi.fn() };
  vi.stubGlobal("WebSocket", vi.fn(function () { return socket; }));
  const client = new QueryClient();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const key = removalKeys.admin([admin.id, admin.role, useAuthStore.getState().sessionVersion]);
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { unmount } = renderHook(() => useAdminNotificationsWs(true), { wrapper });
  await act(async () => socket.onopen());
  expect(invalidate).toHaveBeenCalledWith({ queryKey: key });
  invalidate.mockClear();
  await act(async () => socket.onmessage({ data: JSON.stringify({ type: "photo_removal_changed", entityId: "case" }) }));
  expect(invalidate).toHaveBeenCalledWith({ queryKey: key });
  invalidate.mockClear();
  act(() => { useAuthStore.getState().startSession(admin); socket.onmessage({ data: JSON.stringify({ type: "photo_removal_changed" }) }); });
  expect(invalidate).not.toHaveBeenCalled();
  unmount(); client.clear();
});

it.each(["reconnect", "change"] as const)("cancels an initial removal read without data before refreshing on %s", async (trigger) => {
  const socket = { onopen: () => {}, onmessage: vi.fn<(event: { data: string }) => void>(), close: vi.fn() };
  vi.stubGlobal("WebSocket", vi.fn(function () { return socket; }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const key = [...removalKeys.admin([admin.id, admin.role, useAuthStore.getState().sessionVersion]), "detail", "case"];
  const old = deferred<{ id: string; version: number }>();
  const signals: AbortSignal[] = [];
  const read = vi.fn(({ signal }: { signal: AbortSignal }) => {
    signals.push(signal);
    return signals.length === 1 ? old.promise : Promise.resolve({ id: "case", version: 3 });
  });
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { result, unmount } = renderHook(() => {
    useAdminNotificationsWs(true);
    return useQuery({ queryKey: key, queryFn: read });
  }, { wrapper });
  await waitFor(() => expect(read).toHaveBeenCalledTimes(1));
  expect(client.getQueryData(key)).toBeUndefined();
  await act(async () => {
    if (trigger === "reconnect") socket.onopen();
    else socket.onmessage({ data: JSON.stringify({ type: "photo_removal_changed", entityId: "case" }) });
  });
  expect(signals[0].aborted).toBe(true);
  await waitFor(() => expect(result.current.data).toEqual({ id: "case", version: 3 }));
  expect(read).toHaveBeenCalledTimes(2);
  await act(async () => old.resolve({ id: "case", version: 2 }));
  expect(client.getQueryData(key)).toEqual({ id: "case", version: 3 });
  expect(client.getQueryState(key)?.isInvalidated).toBe(false);
  unmount(); client.clear();
});

it.each(["same-user-login", "logout", "unmount"] as const)("does not invalidate an old removal scope after cancellation finishes following %s", async (transition) => {
  const socket = { onopen: () => {}, onmessage: vi.fn<(event: { data: string }) => void>(), close: vi.fn() };
  vi.stubGlobal("WebSocket", vi.fn(function () { return socket; }));
  const client = new QueryClient();
  const key = removalKeys.admin([admin.id, admin.role, useAuthStore.getState().sessionVersion]);
  const cancelled = deferred<void>();
  const cancel = vi.spyOn(client, "cancelQueries").mockReturnValue(cancelled.promise);
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { unmount } = renderHook(() => useAdminNotificationsWs(true), { wrapper });
  socket.onmessage({ data: JSON.stringify({ type: "photo_removal_changed", entityId: "case" }) });
  expect(cancel).toHaveBeenCalledWith({ queryKey: key });
  act(() => {
    if (transition === "unmount") unmount();
    else if (transition === "logout") useAuthStore.getState().logout();
    else useAuthStore.getState().startSession(admin);
  });
  invalidate.mockClear();
  await act(async () => cancelled.resolve());
  expect(invalidate).not.toHaveBeenCalled();
  unmount(); client.clear();
});
