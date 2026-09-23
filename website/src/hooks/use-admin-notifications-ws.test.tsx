import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import { useAdminNotificationsWs } from "./use-admin-notifications-ws";

vi.mock("@/lib/auth", () => ({ getAccessToken: () => "fixture-token" }));


const admin = { id: "admin-a", role: "ADMIN" as const, name: "Admin", email: "admin@example.test", createdAt: "2026-09-10" };
beforeEach(() => useAuthStore.getState().setUser(admin));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); useAuthStore.getState().logout(); });

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
