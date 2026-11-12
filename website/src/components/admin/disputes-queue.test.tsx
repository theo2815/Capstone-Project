import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Dispute } from "@/lib/admin-disputes";
import { useAuthStore } from "@/store/auth-store";
import { adminDisputeKeys } from "@/hooks/use-admin-disputes";
import { admin, dispute } from "@/test/fixtures/admin";
import { DisputesQueue } from "./disputes-queue";

const fixtures = vi.hoisted(() => ({ search: "", router: { replace: vi.fn() } }));
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/disputes",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

// An old dispute that is not in the loaded list: only the detail endpoint knows it.
const older: Dispute = { ...dispute, id: "dispute-old", runnerHandle: "old-runner" };
const firstPageKey = adminDisputeKeys.page(admin.id, { view: "needs_me", offset: 0 });
const page = (items: Dispute[], total = items.length, offset = 0, limit = 50) => ({ items, total, offset, limit });
const numbered = (n: number): Dispute => ({ ...dispute, id: `dispute-${n}`, runnerHandle: `runner-${n}` });

let client: QueryClient;
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  fixtures.search = "";
  fixtures.router.replace.mockClear();
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockImplementation(async (path) => {
    if (path === `/admin/disputes/${older.id}`) return older;
    if (path.startsWith("/admin/disputes?")) return page([dispute]);
    throw new Error(`unexpected GET ${path}`);
  });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(firstPageKey, page([dispute]));
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  vi.restoreAllMocks();
});

function renderQueue() {
  return render(<QueryClientProvider client={client}><DisputesQueue /></QueryClientProvider>);
}

it("a ?row= link to a dispute outside the loaded list opens it", async () => {
  fixtures.search = `row=${older.id}`;
  renderQueue();
  const drawer = await screen.findByRole("dialog");
  expect(drawer).toHaveTextContent(older.id);
  expect(drawer).toHaveTextContent("@old-runner");
  expect(api.get).toHaveBeenCalledWith(`/admin/disputes/${older.id}`, expect.anything());
  expect(fixtures.router.replace).not.toHaveBeenCalled();
});

it("the drawer holds the admin actions aside", async () => {
  fixtures.search = `row=${dispute.id}`;
  renderQueue();
  const drawer = await screen.findByRole("dialog");
  const aside = within(drawer).getByRole("complementary", { name: "Admin actions" });
  expect(within(aside).getByRole("button", { name: "Resolve…" })).toBeInTheDocument();
  expect(within(aside).getByRole("button", { name: "Deny…" })).toBeInTheDocument();
  expect(within(aside).getByRole("button", { name: "Escalate…" })).toBeInTheDocument();
});

it("Esc on the deny modal closes the modal and leaves the drawer open", async () => {
  const user = userEvent.setup();
  fixtures.search = `row=${dispute.id}`;
  renderQueue();
  const drawer = await screen.findByRole("dialog");
  await user.click(within(drawer).getByRole("button", { name: "Deny…" }));
  expect(screen.getByRole("dialog", { name: `Deny ${dispute.id}` })).toBeInTheDocument();
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog", { name: `Deny ${dispute.id}` })).not.toBeInTheDocument());
  expect(screen.getByRole("dialog")).toHaveTextContent(dispute.id);
  expect(fixtures.router.replace).not.toHaveBeenCalled();
});

it("a failed refund row still offers the decision buttons, so the admin can try the refund again", async () => {
  const failed: Dispute = { ...dispute, status: "open", resolution: "refund_full", refundAmount: 100, refundStatus: "failed" };
  client.setQueryData(firstPageKey, page([failed]));
  fixtures.search = `row=${failed.id}`;
  renderQueue();
  const aside = within(await screen.findByRole("dialog")).getByRole("complementary", { name: "Admin actions" });
  expect(within(aside).getByRole("button", { name: "Resolve…" })).toBeEnabled();
  expect(within(aside).getByRole("button", { name: "Deny…" })).toBeEnabled();
});

it("the queue asks the server for the needs-you view first", async () => {
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  expect(await screen.findByRole("button", { name: `Open dispute ${dispute.id}` })).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&offset=0", expect.anything());
  expect(screen.getByRole("tab", { name: "Needs you" })).toHaveAttribute("aria-selected", "true");
});

it("switching to Closed asks for view=closed and starts at the first page", async () => {
  const user = userEvent.setup();
  vi.mocked(api.get).mockImplementation(async (path) => path.includes("view=closed")
    ? page([{ ...dispute, status: "denied" }])
    : page([numbered(1), numbered(2)], 4, path.includes("offset=2") ? 2 : 0, 2));
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  await user.click(await screen.findByRole("button", { name: "Next →" }));
  await waitFor(() => expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&offset=2", expect.anything()));
  await user.click(screen.getByRole("tab", { name: "Closed" }));
  await waitFor(() => expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=closed&offset=0", expect.anything()));
  expect(screen.getByRole("tab", { name: "Closed" })).toHaveAttribute("aria-selected", "true");
  expect(fixtures.router.replace).toHaveBeenCalledWith("/admin/disputes?view=closed", { scroll: false });
});

it("Next asks for the following page and shows x–y of total", async () => {
  const user = userEvent.setup();
  vi.mocked(api.get).mockImplementation(async (path) => path.includes("offset=2")
    ? page([numbered(3), numbered(4)], 5, 2, 2)
    : page([numbered(1), numbered(2)], 5, 0, 2));
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  expect(await screen.findByText("1–2 of 5")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Next →" }));
  expect(await screen.findByText("3–4 of 5")).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&offset=2", expect.anything());
  expect(screen.getByRole("button", { name: "Open dispute dispute-3" })).toBeInTheDocument();
});

it("search sends q and starts at the first page", async () => {
  const user = userEvent.setup();
  vi.mocked(api.get).mockImplementation(async (path) => path.includes("q=")
    ? page([numbered(9)])
    : page([numbered(1), numbered(2)], 4, path.includes("offset=2") ? 2 : 0, 2));
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  await user.click(await screen.findByRole("button", { name: "Next →" }));
  await user.type(screen.getByRole("searchbox", { name: "Search disputes" }), "  pay_ABC ");
  await waitFor(() => expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&q=pay_ABC&offset=0", expect.anything()));
  expect(await screen.findByRole("button", { name: "Open dispute dispute-9" })).toBeInTheDocument();
});

it("an overdue dispute shows the overdue chip", async () => {
  const late: Dispute = {
    ...dispute, attention: "review", overdue: true,
    dueAt: new Date(Date.now() - 2 * 24 * 3600 * 1000 - 60_000).toISOString(),
  };
  client.setQueryData(firstPageKey, page([late]));
  renderQueue();
  const card = screen.getByRole("button", { name: `Open dispute ${late.id}` });
  expect(card).toHaveTextContent("Overdue · 2d");
  expect(card).toHaveTextContent("Needs review");
});
