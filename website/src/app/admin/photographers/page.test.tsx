import { act, cleanup, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as adminApi from "@/lib/api-admin";
import { useAdminUserActions } from "@/hooks/use-admin-users";
import { useAuthStore } from "@/store/auth-store";
import { admin, adminUser } from "@/test/fixtures/admin";
import AdminPhotographersPage from "./page";

const nav = vi.hoisted(() => ({ search: new URLSearchParams(""), replace: vi.fn(), push: vi.fn() }));
vi.mock("@/lib/api-admin", async (orig) => ({
  ...(await orig<typeof import("@/lib/api-admin")>()),
  fetchAdminUsersPage: vi.fn(), fetchAdminKpis: vi.fn(), unsuspendUser: vi.fn(),
}));
vi.mock("next/navigation", () => ({
  usePathname: () => "/admin/photographers",
  useSearchParams: () => nav.search,
  useRouter: () => ({ replace: nav.replace, push: nav.push }),
}));

const page = (items: typeof adminUser[], total = items.length) => ({ items, total, offset: 0, limit: 50 });
let client: QueryClient;
const Wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
const mount = () => render(<AdminPhotographersPage />, { wrapper: Wrapper });

beforeEach(() => {
  nav.search = new URLSearchParams("");
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.mocked(adminApi.fetchAdminKpis).mockRejectedValue(new Error("no kpis"));
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

it("asks the server for photographers of the tab and search in the URL", async () => {
  nav.search = new URLSearchParams("status=approved&q=ana");
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...adminUser, verificationStatus: "approved" }]));
  mount();
  expect(await screen.findByText("Test Studio")).toBeInTheDocument();
  expect(adminApi.fetchAdminUsersPage).toHaveBeenCalledWith(
    { role: "PHOTOGRAPHER", status: "approved", q: "ana", offset: 0, limit: 50 }, expect.anything());
  expect(screen.getByRole("button", { name: /^Approved/ })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("status")).toHaveTextContent("1 match");
});

it("keeps the tab in the URL", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([adminUser]));
  mount();
  await screen.findByText("Test Studio");
  await userEvent.click(screen.getByRole("button", { name: /^Suspended/ }));
  expect(nav.replace).toHaveBeenCalledWith("/admin/photographers?status=suspended", { scroll: false });
});

it("keeps the Rejected tab in the URL", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([adminUser]));
  mount();
  await screen.findByText("Test Studio");
  await userEvent.click(screen.getByRole("button", { name: /^Rejected/ }));
  expect(nav.replace).toHaveBeenCalledWith("/admin/photographers?status=rejected", { scroll: false });
});

it("lists rejected photographers on their own tab", async () => {
  nav.search = new URLSearchParams("status=rejected");
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...adminUser, verificationStatus: "rejected" }]));
  mount();
  const card = await screen.findByText("Test Studio");
  expect(adminApi.fetchAdminUsersPage).toHaveBeenLastCalledWith(
    { role: "PHOTOGRAPHER", status: "rejected", q: undefined, offset: 0, limit: 50 }, expect.anything());
  expect(screen.getByRole("button", { name: /^Rejected/ })).toHaveAttribute("aria-pressed", "true");
  expect(within(card.closest("a")!).getByText("Rejected")).toBeInTheDocument();
});

it("says so when no photographer is rejected", async () => {
  nav.search = new URLSearchParams("status=rejected");
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([]));
  mount();
  // Reject saves the profile as incomplete today; only older rows carry "rejected".
  expect(await screen.findByText("No rejected photographers. A profile rejected now goes back to Incomplete.")).toBeInTheDocument();
});

it("shows tab counts from the KPIs, and none where no KPI fits", async () => {
  vi.mocked(adminApi.fetchAdminKpis).mockResolvedValue({ pendingVerifications: 3, approvedPhotographers: 12, suspended: 9 } as adminApi.AdminKpis);
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([adminUser]));
  mount();
  await waitFor(() => expect(screen.getByRole("button", { name: /^Pending/ })).toHaveTextContent("Pending3"));
  expect(screen.getByRole("button", { name: /^Approved/ })).toHaveTextContent("Approved12");
  // The suspended KPI counts runners and admins too, so this photographers tab shows no count.
  expect(screen.getByRole("button", { name: /^Suspended/ })).toHaveTextContent(/^Suspended$/);
  expect(screen.getByRole("button", { name: /^Incomplete/ })).toHaveTextContent(/^Incomplete$/);
  expect(screen.getByRole("button", { name: /^Rejected/ })).toHaveTextContent(/^Rejected$/);
});

it("links a photographer with no handle by user id", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...adminUser, handle: null }]));
  mount();
  expect((await screen.findByText("Test Studio")).closest("a")).toHaveAttribute("href", "/admin/photographers/photographer-1");
});

it("pages with Load more", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValueOnce(page([adminUser], 51))
    .mockResolvedValueOnce({ items: [{ ...adminUser, userId: "photographer-2", brandName: "Second Studio" }], total: 51, offset: 1, limit: 50 });
  mount();
  await screen.findByText("Test Studio");
  await userEvent.click(screen.getByRole("button", { name: "Load more" }));
  expect(await screen.findByText("Second Studio")).toBeInTheDocument();
  expect(adminApi.fetchAdminUsersPage).toHaveBeenLastCalledWith(
    { role: "PHOTOGRAPHER", status: "pending", q: undefined, offset: 1, limit: 50 }, expect.anything());
});

it("keeps the paged list working when a user action settles", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...adminUser, suspendedAt: "2026-09-11T00:00:00Z" }]));
  vi.mocked(adminApi.unsuspendUser).mockResolvedValue(adminUser);
  mount();
  await screen.findByText("Test Studio");
  const calls = vi.mocked(adminApi.fetchAdminUsersPage).mock.calls.length;
  const { result } = renderHook(() => useAdminUserActions(), { wrapper: Wrapper });
  await act(async () => { expect(await result.current.unsuspend(adminUser.userId)).toBe(true); });
  await waitFor(() => expect(vi.mocked(adminApi.fetchAdminUsersPage).mock.calls.length).toBeGreaterThan(calls));
  expect(screen.getByText("Test Studio")).toBeInTheDocument();
});

it("keeps the loaded list on screen when a background refresh fails", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValueOnce(page([adminUser])).mockRejectedValue(new Error("Network down"));
  mount();
  await screen.findByText("Test Studio");
  await act(async () => { await client.refetchQueries(); });
  expect(screen.getByText("Test Studio")).toBeInTheDocument();
  expect(screen.queryByText(/Couldn.t load photographers/)).not.toBeInTheDocument();
});

it("never shows one tab's rows under another while the new tab loads", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValueOnce(page([adminUser])).mockReturnValue(new Promise(() => {}));
  const view = mount();
  await screen.findByText("Test Studio");
  nav.search = new URLSearchParams("status=suspended");
  view.rerender(<AdminPhotographersPage />);
  await waitFor(() => expect(screen.getByRole("button", { name: /^Suspended/ })).toHaveAttribute("aria-pressed", "true"));
  expect(screen.queryByText("Test Studio")).not.toBeInTheDocument();
});

it("shows the error and Retry when the first load fails", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockRejectedValue(new Error("Network down"));
  mount();
  expect(await screen.findByText(/Couldn.t load photographers/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
});
