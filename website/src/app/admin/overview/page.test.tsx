import { cleanup, render, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import AdminOverviewPage from "./page";

vi.mock("@/components/admin/admin-overview-trend", () => ({ AdminOverviewTrend: () => null }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => undefined }));

afterEach(() => { cleanup(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("asks for the 20 most recent decisions", async () => {
  useAuthStore.getState().setUser(admin);
  const get = vi.spyOn(api, "get").mockResolvedValue([]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>);
  await waitFor(() => expect(get).toHaveBeenCalledWith("/admin/decisions?limit=20", expect.anything()));
});
