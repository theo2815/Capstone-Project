import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import AdminOverviewPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

vi.mock("@/components/admin/admin-overview-trend", () => ({ AdminOverviewTrend: () => null }));
const kpis = vi.hoisted(() => ({ value: undefined as unknown, isError: false, refetch: () => {} }));
vi.mock("@/hooks/use-admin-data", () => ({
  useAdminKpisQuery: () => ({ data: kpis.value, isError: kpis.isError, isFetching: false, refetch: kpis.refetch }),
}));

afterEach(() => { kpis.value = undefined; kpis.isError = false; cleanup(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

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

it("the decisions tile links to the decision history on the same page", () => {
  useAuthStore.getState().setUser(admin);
  kpis.value = { decisionsThisWeek: 3 };
  vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>);
  expect(screen.getByText(/This week's decisions/).closest("a")?.getAttribute("href")).toBe("#decisions");
});

it("shows a retry instead of zero counts when the metrics fail to load", async () => {
  useAuthStore.getState().setUser(admin);
  kpis.isError = true;
  kpis.refetch = vi.fn();
  vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>);
  expect(screen.getByText("Couldn't load the metrics. Please try again.")).toBeInTheDocument();
  expect(screen.queryByText(/Disputes needing you/)).toBeNull();
  screen.getByRole("button", { name: "Retry" }).click();
  expect(kpis.refetch).toHaveBeenCalled();
});

// jsdom has no layout: the browser check proves the fit; this pins the rule that gives it.
it("the tiles and their loading grid take their columns from the room beside the rail, not the window", () => {
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const page = () => <QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>;
  const { container, rerender } = render(page());
  const loading = container.querySelector('[aria-busy="true"]')!;
  expect(loading.parentElement!.classList.contains("@container")).toBe(true);
  expect(loading.className).not.toMatch(/(^|\s)[\w[\].-]+:grid-cols-/); // any window breakpoint; @lg: etc. pass
  const loadingClass = loading.className;

  kpis.value = { pendingVerifications: 2 };
  rerender(page());
  const label = screen.getByText("Pending verifications");
  const grid = label.closest("a")!.parentElement!;
  expect(grid.className).toBe(loadingClass);
  expect(grid.parentElement!.classList.contains("@container")).toBe(true);
  // Two whole lines of the label's own line height, so a one-line label's value starts level with a two-line one's.
  expect(label.className).toContain("min-h-[2lh]");
});

it("the heading is the rail label", () => {
  vi.spyOn(api, "get").mockReturnValue(new Promise(() => {}));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Overview.");
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  vi.spyOn(api, "get").mockReturnValue(new Promise(() => {}));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdminOverviewPage /></QueryClientProvider>);
  expectSectionIntro(ROUTES.ADMIN_OVERVIEW);
});
