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

let client: QueryClient;
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  fixtures.search = "";
  fixtures.router.replace.mockClear();
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockImplementation(async (path) => {
    if (path === `/admin/disputes/${older.id}`) return older;
    if (path.startsWith("/admin/disputes?")) return { items: [dispute], total: 1, offset: 0, limit: 200 };
    throw new Error(`unexpected GET ${path}`);
  });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(adminDisputeKeys.list(admin.id), [dispute]);
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
  client.setQueryData(adminDisputeKeys.list(admin.id), [failed]);
  fixtures.search = `row=${failed.id}`;
  renderQueue();
  const aside = within(await screen.findByRole("dialog")).getByRole("complementary", { name: "Admin actions" });
  expect(within(aside).getByRole("button", { name: "Resolve…" })).toBeEnabled();
  expect(within(aside).getByRole("button", { name: "Deny…" })).toBeEnabled();
});
