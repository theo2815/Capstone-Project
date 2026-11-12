import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import AdminOverviewPage from "./page";

vi.mock("@/components/admin/admin-overview-trend", () => ({ AdminOverviewTrend: () => null }));
const kpis = vi.hoisted(() => ({ value: undefined as unknown }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => kpis.value }));

afterEach(() => { kpis.value = undefined; cleanup(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("asks for the 20 most recent decisions", async () => {
  useAuthStore.getState().setUser(admin);
  const get = vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>);
  await waitFor(() => expect(get).toHaveBeenCalledWith("/admin/decisions?limit=20", expect.anything()));
});

it("the disputes tile counts everything that needs an admin", async () => {
  useAuthStore.getState().setUser(admin);
  kpis.value = { openDisputes: 1, disputesNeedingAdmin: 4, disputesOverdue: 2 };
  vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>);
  const tile = screen.getByText(/Disputes needing you/).closest("a")!;
  expect(tile.textContent).toContain("4");
  expect(tile.textContent).toContain("2 overdue");
});
