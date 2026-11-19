import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { admin } from "@/test/fixtures/admin";
import { useRefundedThisWeekTotal } from "./use-admin-disputes";

let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  client.clear();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
});

it("refunded this week comes from the sales KPIs, not a sum of the dispute list", async () => {
  vi.spyOn(api, "get").mockImplementation(async (path) => {
    if (path === "/admin/sales/kpis?range=week") return { gmv: 2500, platformRevenue: 625, refundsIssued: 340, netPlatformRevenue: 285, photographerKeep: 1875, totalSalesCount: 25 };
    // A list that a sum would read as 80 refunded.
    return [{ id: "d1", status: "resolved", refundAmount: 80, resolvedAt: new Date().toISOString() }];
  });
  const { result } = renderHook(() => useRefundedThisWeekTotal(), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
  await waitFor(() => expect(result.current).toBe(340));
  expect(api.get).toHaveBeenCalledWith("/admin/sales/kpis?range=week", expect.anything());
});
