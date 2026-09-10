import { cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { useAdminNotificationsWs } from "./use-admin-notifications-ws";

vi.mock("@/lib/auth", () => ({ getAccessToken: () => "fixture-token" }));
vi.mock("@/lib/admin-users-data", () => ({
  useAdminUsersServerStore: { getState: () => ({ invalidate: vi.fn(), refetch: vi.fn() }) },
}));

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("refreshes canonical queues on reconnect and filed events, and ignores messages after teardown", () => {
  const socket = { onopen: () => {}, onmessage: vi.fn<(event: { data: string }) => void>(), close: vi.fn() };
  vi.stubGlobal("WebSocket", vi.fn(function () { return socket; }));
  const client = new QueryClient();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { unmount } = renderHook(() => useAdminNotificationsWs(true), { wrapper });
  socket.onopen();
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "disputes"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "payouts"] });
  invalidate.mockClear();
  socket.onmessage({ data: JSON.stringify({ type: "dispute_filed" }) });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "disputes"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "kpis"] });
  socket.onmessage({ data: JSON.stringify({ type: "payout_report_filed" }) });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "payouts", "reports"] });
  unmount();
  invalidate.mockClear();
  socket.onmessage({ data: JSON.stringify({ type: "dispute_filed" }) });
  expect(invalidate).not.toHaveBeenCalled();
  client.clear();
});
