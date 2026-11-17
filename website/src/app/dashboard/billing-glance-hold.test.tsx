import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import DashboardOverviewPage from "./page";

vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => ({ kind: "ok" }) }));
vi.mock("@/components/dashboard/dashboard-action-grid", () => ({ DashboardActionGrid: () => null }));

const balance = {
  unpaidBalance: 1000, clearing: 0, reserved: 0, minimum: 500,
  hasOpenRequest: false, openRequest: null, accountCooldownUntil: null,
  primaryAccount: null,
};

let client: QueryClient;
function setup(payoutHoldReason: string | null) {
  useAuthStore.getState().setUser({ id: "seller-1", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.includes("/payouts/balance")) return Promise.resolve({ ...balance, payoutHoldReason });
    return Promise.resolve({ items: [{ id: "event-1", name: "Cebu Run", date: "2026-11-01", state: "past", photoCount: 3 }], total: 1, offset: 0, limit: 8 });
  });
  render(<QueryClientProvider client={client}><DashboardOverviewPage /></QueryClientProvider>);
}

afterEach(() => { cleanup(); client?.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("the billing card says payouts are on hold instead of ready to request", async () => {
  setup("UPLOAD_REVIEW");
  expect(await screen.findByText("Payouts on hold — open billing for details.")).toBeVisible();
  expect(screen.queryByText(/Ready to request/)).not.toBeInTheDocument();
});

it("the billing card says ready to request without a hold", async () => {
  setup(null);
  expect(await screen.findByText(/Ready to request/)).toBeVisible();
  expect(screen.queryByText(/Payouts on hold/)).not.toBeInTheDocument();
});
