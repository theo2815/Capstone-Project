import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import DashboardOverviewPage from "./page";

vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => ({ kind: "ok" }) }));
vi.mock("@/components/dashboard/dashboard-action-grid", () => ({ DashboardActionGrid: () => null }));

let client: QueryClient;
afterEach(() => { cleanup(); client?.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("Next up lists the live race first, with an Upload link, then upcoming races by date", async () => {
  useAuthStore.getState().setUser({ id: "p-1", role: "PHOTOGRAPHER", name: "Jessa", email: "p@example.test", createdAt: "2026-09-01" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.includes("/payouts/balance")) return Promise.resolve({ unpaidBalance: 0, clearing: 0, reserved: 0, minimum: 500, hasOpenRequest: false, openRequest: null, accountCooldownUntil: null, primaryAccount: null });
    return Promise.resolve({ items: [
      { id: "later", name: "Later Run", date: "2026-12-20", state: "upcoming", photoCount: 0 },
      { id: "past", name: "Old Run", date: "2026-08-01", state: "past", photoCount: 9 },
      { id: "live", name: "Live Trail", date: "2026-11-15", state: "live", photoCount: 4 },
      { id: "soon", name: "Soon Run", date: "2026-11-29", state: "upcoming", photoCount: 0 },
    ], total: 4, offset: 0, limit: 8 });
  });
  render(<QueryClientProvider client={client}><DashboardOverviewPage /></QueryClientProvider>);
  const nextUp = (await screen.findByRole("heading", { name: "Next up" })).closest("section")!;
  const rows = within(nextUp).getAllByRole("heading", { level: 3 }).map((h) => h.textContent);
  expect(rows).toEqual(["Live Trail", "Soon Run", "Later Run"]);
  expect(within(nextUp).getByRole("link", { name: /Live Trail.*Upload/ })).toHaveAttribute("href", "/upload/live");
  expect(await screen.findByText(/reach ₱500\.00 to request a payout/)).toBeInTheDocument();
});

it("Next up leaves out the photographer's own race that still waits for review", async () => {
  useAuthStore.getState().setUser({ id: "p-1", role: "PHOTOGRAPHER", name: "Jessa", email: "p@example.test", createdAt: "2026-09-01" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.includes("/payouts/balance")) return Promise.resolve({ unpaidBalance: 0, clearing: 0, reserved: 0, minimum: 500, hasOpenRequest: false, openRequest: null, accountCooldownUntil: null, primaryAccount: null });
    return Promise.resolve({ items: [
      { id: "waiting", name: "Waiting Run", date: "2026-11-18", state: "live", photoCount: 0, ownedByMe: true, reviewStatus: "pending" },
      { id: "approved", name: "Approved Run", date: "2026-11-18", state: "live", photoCount: 0, ownedByMe: true, reviewStatus: "approved" },
      { id: "live", name: "Live Trail", date: "2026-11-18", state: "live", photoCount: 4 },
    ], total: 3, offset: 0, limit: 8 });
  });
  render(<QueryClientProvider client={client}><DashboardOverviewPage /></QueryClientProvider>);
  const nextUp = (await screen.findByRole("heading", { name: "Next up" })).closest("section")!;
  const rows = within(nextUp).getAllByRole("heading", { level: 3 }).map((h) => h.textContent);
  expect(rows).toEqual(["Approved Run", "Live Trail"]);
});

it("a failed events read says so and retries, instead of loading forever", async () => {
  useAuthStore.getState().setUser({ id: "p-1", role: "PHOTOGRAPHER", name: "Jessa", email: "p@example.test", createdAt: "2026-09-01" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const get = vi.spyOn(api, "get").mockRejectedValue(new Error("boom"));
  render(<QueryClientProvider client={client}><DashboardOverviewPage /></QueryClientProvider>);
  expect(await screen.findByText("Couldn't load your dashboard.")).toBeInTheDocument();
  get.mockResolvedValue({ items: [], total: 0, offset: 0, limit: 8 });
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  await waitFor(() => expect(screen.queryByText("Couldn't load your dashboard.")).not.toBeInTheDocument());
  expect(get).toHaveBeenCalledTimes(2);
});
