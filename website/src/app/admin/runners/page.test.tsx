import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as adminApi from "@/lib/api-admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { admin, adminUser } from "@/test/fixtures/admin";
import AdminRunnersPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

const nav = vi.hoisted(() => ({ search: new URLSearchParams(""), replace: vi.fn(), push: vi.fn() }));
vi.mock("@/lib/api-admin", async (orig) => ({
  ...(await orig<typeof import("@/lib/api-admin")>()),
  fetchAdminUsersPage: vi.fn(), suspendUser: vi.fn(), unsuspendUser: vi.fn(),
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
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); useToastStore.setState({ toasts: [] }); vi.resetAllMocks(); });

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

// FU-18: runner= keeps only this runner's disputes; q= keeps today's email search for a backend without the filter.
it("links each runner to their own disputes, by id, named for the runner", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...runner, email: "ana+race@example.test" }]));
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  const link = within(row).getByRole("link", { name: "Disputes of Ana Runner" });
  expect(link).toHaveTextContent("Their disputes →");
  expect(link).toHaveAttribute("href", "/admin/disputes?view=all&q=ana%2Brace%40example.test&runner=runner-7");
});

it("lists all runners by default and counts both tabs", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockImplementation(async (args) =>
    args.limit === 1 ? { items: [], total: args.status === "suspended" ? 2 : 9, offset: 0, limit: 1 } : page([runner]));
  mount();
  expect(await screen.findByText("Ana Runner")).toBeInTheDocument();
  expect(adminApi.fetchAdminUsersPage).toHaveBeenCalledWith({ role: "RUNNER", offset: 0, limit: 50 }, expect.anything());
  const tabs = screen.getByRole("group", { name: "Runner status" });
  expect(await within(tabs).findByRole("button", { name: /^All\D*9/ })).toHaveAttribute("aria-pressed", "true");
  expect(await within(tabs).findByRole("button", { name: /^Suspended\D*2/ })).toHaveAttribute("aria-pressed", "false");
});

it("the Suspended tab asks the server for suspended runners only", async () => {
  nav.search = new URLSearchParams("status=suspended");
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([]));
  mount();
  expect(await screen.findByText("No active suspensions.")).toBeInTheDocument();
  expect(adminApi.fetchAdminUsersPage).toHaveBeenCalledWith(
    { role: "RUNNER", status: "suspended", offset: 0, limit: 50 }, expect.anything());
});

it("an unsuspended runner leaves the Suspended tab at once, before the list refetch lands", async () => {
  nav.search = new URLSearchParams("status=suspended");
  const held = { ...runner, suspendedAt: "2026-11-01T00:00:00Z" };
  vi.mocked(adminApi.fetchAdminUsersPage)
    .mockImplementation(async (args) => args.limit === 1 ? { items: [], total: 1, offset: 0, limit: 1 } : page([held]));
  vi.mocked(adminApi.unsuspendUser).mockResolvedValue({ ...runner, suspendedAt: null });
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  vi.mocked(adminApi.fetchAdminUsersPage).mockReturnValue(new Promise(() => {})); // the refetch is held
  await userEvent.click(within(row).getByRole("button", { name: "Unsuspend" }));
  await waitFor(() => expect(screen.queryByText("Ana Runner")).toBeNull());
});

it("picking a tab writes it to the URL", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([runner]));
  mount();
  await userEvent.click(within(screen.getByRole("group", { name: "Runner status" })).getByRole("button", { name: /^Suspended/ }));
  expect(nav.replace).toHaveBeenCalledWith("/admin/runners?status=suspended", { scroll: false });
});

it("suspends the runner whose row opened the shared Suspend modal", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...runner, userId: "runner-1", name: "Other Runner" }, runner]));
  vi.mocked(adminApi.suspendUser).mockResolvedValue({ ...runner, suspendedAt: "2026-11-20T00:00:00Z" });
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  await userEvent.click(within(row).getByRole("button", { name: /Suspend/ }));
  const dialog = await screen.findByRole("dialog", { name: "Suspend account" });
  expect(dialog).toHaveTextContent("Ana Runner is signed out shortly and cannot sign in until you unsuspend. The reason is saved for them, but they cannot sign in to read it and QuickPitik does not email them, so tell them yourself if they should know.");
  expect(dialog).not.toHaveTextContent("They get a message with the reason");
  await userEvent.click(within(dialog).getByLabelText("Compliance review requested"));
  await userEvent.click(within(dialog).getByRole("button", { name: "Suspend" }));
  await waitFor(() => expect(adminApi.suspendUser).toHaveBeenCalledWith("runner-7", "Compliance review requested"));
  await waitFor(() => expect(useToastStore.getState().toasts[0]).toMatchObject({ kind: "success", message: "Ana Runner is suspended. They were not emailed." }));
});

it("offers the runner suspend reasons, not the photographer ones (AX-72)", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([runner]));
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  await userEvent.click(within(row).getByRole("button", { name: /Suspend/ }));
  const dialog = await screen.findByRole("dialog", { name: "Suspend account" });
  expect(within(dialog).getByLabelText("Using someone else's selfie or account")).toBeInTheDocument();
  expect(within(dialog).queryByLabelText("Watermark or branding violates policy")).toBeNull();
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
  expect(dialog).toHaveTextContent("Write to Ana Runner privately. The message goes to their QuickPitik inbox. No email is sent.");
  expect(within(dialog).getByLabelText(/^Subject/)).toHaveValue("");
  expect(within(dialog).getByLabelText(/^Subject/)).toHaveAttribute("placeholder", "About your refund request");
  expect(within(dialog).getByLabelText(/^Message/)).toHaveAttribute("placeholder", "Hi, this is QuickPitik support.");
});

it("offers Unsuspend for a suspended runner", async () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([{ ...runner, suspendedAt: "2026-11-01T00:00:00Z" }]));
  mount();
  const row = (await screen.findByText("Ana Runner")).closest("li")!;
  expect(within(row).getByText("Suspended")).toBeInTheDocument();
  expect(within(row).getByRole("button", { name: "Unsuspend" })).toBeInTheDocument();
  expect(within(row).getByRole("button", { name: /Send message/ })).toBeInTheDocument();
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  vi.mocked(adminApi.fetchAdminUsersPage).mockResolvedValue(page([]));
  mount();
  expectSectionIntro(ROUTES.ADMIN_RUNNERS);
});
