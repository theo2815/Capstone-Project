import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useAdminKpis } from "./use-admin-data";

beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
afterEach(() => { cleanup(); vi.useRealTimers(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("the KPIs refetch on their own after the poll interval", async () => {
  useAuthStore.getState().setUser(admin);
  const get = vi.spyOn(api, "get").mockResolvedValue({ openDisputes: 0 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  renderHook(() => useAdminKpis(), { wrapper });
  await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });
  const first = get.mock.calls.length;
  expect(first).toBeGreaterThan(0);
  await act(async () => { await vi.advanceTimersByTimeAsync(61_000); });
  expect(get.mock.calls.length).toBeGreaterThan(first);
});
