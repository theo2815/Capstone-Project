import { cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useAdminWsStatusStore } from "@/store/admin-ws-status-store";
import { useAdminQueueRealtime } from "./use-admin-queue-realtime";

beforeEach(() => { vi.useFakeTimers(); useAuthStore.getState().setUser(admin); useAdminWsStatusStore.getState().setStatus("degraded"); });
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("the polling fallback refreshes the relevance cases and alerts", () => {
  const client = new QueryClient();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { unmount } = renderHook(() => useAdminQueueRealtime(true), { wrapper });
  vi.advanceTimersByTime(29_999);
  expect(invalidate).not.toHaveBeenCalled();
  vi.advanceTimersByTime(1);
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "relevance"] });
  unmount(); client.clear();
});
