import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as adminApi from "@/lib/api-admin";
import { useAuthStore } from "@/store/auth-store";
import { admin, adminUser } from "@/test/fixtures/admin";
import AdminRunnersPage from "./page";

const nav = vi.hoisted(() => ({ search: new URLSearchParams(""), replace: vi.fn(), push: vi.fn() }));
vi.mock("@/lib/api-admin", async (orig) => ({
  ...(await orig<typeof import("@/lib/api-admin")>()),
  fetchAdminUsersPage: vi.fn(), suspendUser: vi.fn(),
}));
vi.mock("next/navigation", () => ({
  usePathname: () => "/admin/runners",
  useSearchParams: () => nav.search,
  useRouter: () => ({ replace: nav.replace, push: nav.push }),
}));

const runner = {
  ...adminUser, userId: "runner-7", role: "RUNNER" as const, name: "Ana Runner", email: "ana@example.test",
  brandName: null, handle: null, settingsSnapshot: null,
};
const page = (items: typeof runner[]) => ({ items, total: items.length, offset: 0, limit: 50 });
let client: QueryClient;
const mount = () => render(<AdminRunnersPage />, {
  wrapper: ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
});

beforeEach(() => {
  nav.search = new URLSearchParams("");
  useAuthStore.getState().startSession(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

it("searches runners on the server with the search in the URL", async () => {
  nav.search = new URLSearchParams("q=ana");
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([runner]));
  mount();
  expect(await screen.findByText("Ana Runner")).toBeInTheDocument();
  expect(adminApi.fetchAdminUsersPage).toHaveBeenCalledWith(
    { role: "RUNNER", q: "ana", offset: 0, limit: 50 }, expect.anything());
  expect(screen.getByText("ana@example.test")).toBeInTheDocument();
  expect(screen.getByText("Active")).toBeInTheDocument();
});

it("suspends the runner whose row opened the shared Suspend modal", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...runner, userId: "runner-1", name: "Other Runner" }, runner]));
  vi.mocked(adminApi.suspendUser).mockResolvedValue({ ...runner, suspendedAt: "2026-11-20T00:00:00Z" });
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  await userEvent.click(within(row).getByRole("button", { name: /Suspend/ }));
  const dialog = await screen.findByRole("dialog", { name: "Suspend account" });
  expect(dialog).toHaveTextContent("Ana Runner will be signed out");
  await userEvent.click(within(dialog).getByLabelText("Compliance review requested"));
  await userEvent.click(within(dialog).getByRole("button", { name: "Suspend" }));
  await waitFor(() => expect(adminApi.suspendUser).toHaveBeenCalledWith("runner-7", "Compliance review requested"));
});

it("turns the row to Unsuspend at once after a suspend, before the list refetch lands", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage)
    .mockResolvedValueOnce(page([runner]))
    .mockReturnValue(new Promise(() => {})); // the refetch is held
  vi.mocked(adminApi.suspendUser).mockResolvedValue({ ...runner, suspendedAt: "2026-11-20T00:00:00Z" });
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  await userEvent.click(within(row).getByRole("button", { name: /Suspend/ }));
  const dialog = await screen.findByRole("dialog", { name: "Suspend account" });
  await userEvent.click(within(dialog).getByLabelText("Compliance review requested"));
  await userEvent.click(within(dialog).getByRole("button", { name: "Suspend" }));
  expect(await within(row).findByRole("button", { name: "Unsuspend" })).toBeInTheDocument();
  expect(within(row).queryByRole("button", { name: /Suspend…/ })).toBeNull();
  expect(adminApi.suspendUser).toHaveBeenCalledTimes(1);
});

it("never carries a message draft from one runner to another", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...runner, userId: "runner-1", name: "Other Runner" }, runner]));
  mount();
  const rows = (await screen.findAllByRole("listitem"));
  await userEvent.click(within(rows[0]).getByRole("button", { name: /Send message/ }));
  await userEvent.type(await screen.findByLabelText(/^Subject/), "For Other only");
  await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await userEvent.click(within(rows[1]).getByRole("button", { name: /Send message/ }));
  const dialog = await screen.findByRole("dialog", { name: "Send message" });
  expect(dialog).toHaveTextContent("Reach out to Ana Runner privately. The message lands directly in their inbox.");
  expect(within(dialog).getByLabelText(/^Subject/)).toHaveValue("");
});

it("offers Unsuspend for a suspended runner", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...runner, suspendedAt: "2026-11-01T00:00:00Z" }]));
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  expect(within(row).getByText("Suspended")).toBeInTheDocument();
  expect(within(row).getByRole("button", { name: "Unsuspend" })).toBeInTheDocument();
  expect(within(row).getByRole("button", { name: /Send message/ })).toBeInTheDocument();
});
