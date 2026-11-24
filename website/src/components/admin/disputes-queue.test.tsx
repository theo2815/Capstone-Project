import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Dispute } from "@/lib/admin-disputes";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { adminDisputeKeys } from "@/hooks/use-admin-disputes";
import { admin, dispute } from "@/test/fixtures/admin";
import { DisputesQueue } from "./disputes-queue";

const fixtures = vi.hoisted(() => ({ search: "", router: { replace: vi.fn() }, kpis: {} as Record<string, number> }));
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/disputes",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

// An old dispute that is not in the loaded list: only the detail endpoint knows it.
const older: Dispute = { ...dispute, id: "dispute-old", runnerHandle: "old-runner" };
const firstPageKey = adminDisputeKeys.page(admin.id, { view: "needs_me" });
const page = (items: Dispute[], total = items.length, offset = 0, limit = 50) => ({ items, total, offset, limit });
const numbered = (n: number): Dispute => ({ ...dispute, id: `dispute-${n}`, runnerHandle: `runner-${n}` });
// The queue reads the server pages through useInfiniteList, so the cache holds {pages, pageParams}.
const seed = (items: Dispute[]) => client.setQueryData(firstPageKey, { pages: [page(items)], pageParams: [0] });

let client: QueryClient;
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  fixtures.search = "";
  fixtures.router.replace.mockClear();
  fixtures.kpis = {};
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockImplementation(async (path) => {
    if (path === `/admin/disputes/${older.id}`) return older;
    if (path.startsWith("/admin/disputes?")) return page([dispute]);
    if (path === "/admin/kpis") return fixtures.kpis;
    throw new Error(`unexpected GET ${path}`);
  });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  seed([dispute]);
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
  expect(within(aside).getByRole("button", { name: "Approve refund…" })).toBeInTheDocument();
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
  seed([failed]);
  fixtures.search = `row=${failed.id}`;
  renderQueue();
  const aside = within(await screen.findByRole("dialog")).getByRole("complementary", { name: "Admin actions" });
  expect(within(aside).getByRole("button", { name: "Approve refund…" })).toBeEnabled();
  expect(within(aside).getByRole("button", { name: "Deny…" })).toBeEnabled();
});

it("the queue asks the server for the needs-you view first", async () => {
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  expect(await screen.findByRole("button", { description: dispute.id })).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&offset=0", expect.anything());
  expect(screen.getByRole("button", { name: /^Needs you/ })).toHaveAttribute("aria-pressed", "true");
  // One page only: no Load more row and no "All 1 loaded" line, as with the old pager.
  expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  expect(screen.queryByText(/loaded/)).not.toBeInTheDocument();
});

it("switching to Closed asks for view=closed and starts at the first page", async () => {
  const user = userEvent.setup();
  vi.mocked(api.get).mockImplementation(async (path) => path.includes("view=closed")
    ? page([{ ...dispute, status: "denied" }])
    : page([numbered(1), numbered(2)], 54, path.includes("offset=2") ? 2 : 0, 2));
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  await user.click(await screen.findByRole("button", { name: "Load more" }));
  await waitFor(() => expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&offset=2", expect.anything()));
  await user.click(screen.getByRole("button", { name: "Closed" }));
  await waitFor(() => expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=closed&offset=0", expect.anything()));
  expect(screen.getByRole("button", { name: "Closed" })).toHaveAttribute("aria-pressed", "true");
  expect(fixtures.router.replace).toHaveBeenCalledWith("/admin/disputes?view=closed", { scroll: false });
});

it("Load more appends the following server page under the first", async () => {
  const user = userEvent.setup();
  vi.mocked(api.get).mockImplementation(async (path) => path.includes("offset=2")
    ? page([numbered(3), numbered(4)], 55, 2, 2)
    : page([numbered(1), numbered(2)], 55, 0, 2));
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  expect(await screen.findByText("Showing 2 of 55")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Load more" }));
  expect(await screen.findByText("Showing 4 of 55")).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&offset=2", expect.anything());
  expect(screen.getByRole("button", { description: "dispute-1" })).toBeInTheDocument();
  expect(screen.getByRole("button", { description: "dispute-3" })).toBeInTheDocument();
});

it("a failed refresh with rows on screen says so and offers Retry", async () => {
  const user = userEvent.setup();
  seed([dispute]);
  renderQueue();
  expect(screen.getByRole("button", { description: dispute.id })).toBeInTheDocument();
  vi.mocked(api.get).mockRejectedValue(new Error("offline"));
  await act(() => client.refetchQueries({ queryKey: firstPageKey }));
  const notice = await screen.findByRole("alert");
  expect(notice).toHaveTextContent("Couldn't refresh.");
  expect(screen.getByRole("button", { description: dispute.id })).toBeInTheDocument();
  vi.mocked(api.get).mockResolvedValue(page([dispute]));
  await user.click(within(notice).getByRole("button", { name: "Retry" }));
  await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
});

it("the view tabs count from the KPIs, and show no count when a field is missing", async () => {
  fixtures.kpis = { disputesNeedingAdmin: 7, disputesOverdue: 2 };
  seed([dispute]);
  renderQueue();
  expect(await screen.findByRole("button", { name: "Needs you 7" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Overdue 2" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Closed" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "All" })).toBeInTheDocument();
  cleanup();
  client.clear();
  fixtures.kpis = { openDisputes: 4 };
  seed([dispute]);
  renderQueue();
  expect(await screen.findByRole("button", { name: "Needs you 4" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Overdue" })).toBeInTheDocument();
});

it("on a phone the actions scroll with the drawer instead of a capped footer", async () => {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: false, media: query, addEventListener: () => {}, removeEventListener: () => {} }));
  try {
    const drawer = await openDrawerFor(dispute);
    expect(drawer.querySelector("footer")).toBeNull();
    expect(within(drawer).getByRole("complementary", { name: "Admin actions" })).toBeInTheDocument();
  } finally {
    vi.unstubAllGlobals();
  }
});

it("on a phone the action buttons sit in a row stuck to the bottom of the drawer, and each opens its modal", async () => {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: false, media: query, addEventListener: () => {}, removeEventListener: () => {} }));
  try {
    const user = userEvent.setup();
    const drawer = await openDrawerFor(dispute);
    const row = within(drawer).getByRole("group", { name: "Dispute actions" });
    expect(row).toHaveClass("sticky", "bottom-0");
    // The buttons move: the card at the top keeps no copy of them.
    expect(within(within(drawer).getByRole("complementary", { name: "Admin actions" })).queryAllByRole("button")).toHaveLength(0);
    // The row comes after the evidence, so it can stay in view to the end of the scroll.
    const claim = within(drawer).getByText(dispute.note);
    expect(claim.compareDocumentPosition(row) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    for (const [button, modal] of [["Approve refund…", `Approve refund · ${dispute.id}`], ["Deny…", `Deny ${dispute.id}`], ["Escalate…", `Escalate ${dispute.id}`]]) {
      await user.click(within(row).getByRole("button", { name: button }));
      expect(await screen.findByRole("dialog", { name: modal })).toBeInTheDocument();
      await user.keyboard("{Escape}");
      await waitFor(() => expect(screen.queryByRole("dialog", { name: modal })).toBeNull());
    }
  } finally {
    vi.unstubAllGlobals();
  }
});

it("on a wider screen the actions stay in the drawer footer", async () => {
  const drawer = await openDrawerFor(dispute);
  const footer = drawer.querySelector("footer")!;
  expect(within(footer).getByRole("complementary", { name: "Admin actions" })).toBeInTheDocument();
  expect(within(footer).getByRole("button", { name: "Approve refund…" })).toBeInTheDocument();
  expect(within(drawer).queryByRole("group", { name: "Dispute actions" })).toBeNull();
});

it("search sends q and starts at the first page", async () => {
  const user = userEvent.setup();
  vi.mocked(api.get).mockImplementation(async (path) => path.includes("q=")
    ? page([numbered(9)])
    : page([numbered(1), numbered(2)], 54, path.includes("offset=2") ? 2 : 0, 2));
  client.removeQueries({ queryKey: firstPageKey });
  renderQueue();
  await user.click(await screen.findByRole("button", { name: "Load more" }));
  await user.type(screen.getByRole("searchbox", { name: "Search disputes" }), "  pay_ABC ");
  await waitFor(() => expect(api.get).toHaveBeenCalledWith("/admin/disputes?view=needs_me&q=pay_ABC&offset=0", expect.anything()));
  expect(await screen.findByRole("button", { description: "dispute-9" })).toBeInTheDocument();
});

it("an overdue dispute shows the overdue chip", async () => {
  const late: Dispute = {
    ...dispute, attention: "review", overdue: true,
    dueAt: new Date(Date.now() - 2 * 24 * 3600 * 1000 - 60_000).toISOString(),
  };
  seed([late]);
  renderQueue();
  const card = screen.getByRole("button", { description: late.id }).closest("article")!;
  expect(card).toHaveTextContent("Overdue · 2d");
  // FT-18 (C-40): "Needs review" only repeats the Open pill, so it is not shown.
  expect(card).not.toHaveTextContent("Needs review");
});

it("Esc on the Deny opened from Approve closes Deny and leaves the drawer open", async () => {
  const user = userEvent.setup();
  const blocked: Dispute = { ...dispute, refundable: false, refundBlockedReason: "window_closed", refundBlockedMessage: "Window closed." };
  seed([blocked]);
  vi.mocked(api.get).mockImplementation(async (path) => {
    if (path === "/admin/disputes/deny-reasons") {
      return page([{ code: "window_closed", label: "Window closed", runnerText: "Window closed.", noteRequired: false }] as never);
    }
    return page([blocked]);
  });
  fixtures.search = `row=${blocked.id}`;
  renderQueue();
  const drawer = await screen.findByRole("dialog");
  await user.click(within(drawer).getByRole("button", { name: "Approve refund…" }));
  await user.click(within(screen.getByRole("dialog", { name: `Approve refund · ${blocked.id}` })).getByRole("button", { name: "Deny with this reason" }));
  expect(screen.getByRole("dialog", { name: `Deny ${blocked.id}` })).toBeInTheDocument();
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog", { name: `Deny ${blocked.id}` })).not.toBeInTheDocument());
  expect(screen.getByRole("dialog")).toHaveTextContent(blocked.id);
  expect(fixtures.router.replace).not.toHaveBeenCalled();
});

async function openDrawerFor(d: Dispute) {
  seed([d]);
  fixtures.search = `row=${d.id}`;
  renderQueue();
  return screen.findByRole("dialog");
}

it("the drawer shows downloads started and the runner's other requests", async () => {
  const drawer = await openDrawerFor({
    ...dispute, itemChargeAmount: 110, paymentPhotoCount: 3, refundable: true,
    downloadCount: 4, lastDownloadedAt: "2026-09-09T05:00:00Z",
    runnerHistory: { total: 5, refunded: 2, denied: 1, withdrawn: 1, open: 1 },
  });
  expect(within(drawer).getByText("This photo's charge").nextSibling).toHaveTextContent("110");
  expect(within(drawer).getByText("Payment covered").nextSibling).toHaveTextContent("3 photos");
  expect(within(drawer).getByText("Can it be refunded?").nextSibling).toHaveTextContent("Yes");
  expect(within(drawer).getByText("Downloads started").nextSibling).toHaveTextContent("4");
  expect(drawer).toHaveTextContent("Counts download starts; retries and share attempts count again.");
  const other = within(drawer).getByText("This runner's other requests").nextSibling;
  expect(other).toHaveTextContent("5 total · 2 refunded · 1 declined · 1 withdrawn · 1 open");
});

it("a dispute with no downloads says so", async () => {
  const none = await openDrawerFor({ ...dispute, downloadCount: 0 });
  expect(within(none).getByText("Downloads started").nextSibling).toHaveTextContent("No downloads yet");
  cleanup();
  const noRecord = await openDrawerFor({ ...dispute, downloadCount: null });
  expect(within(noRecord).getByText("Downloads started").nextSibling).toHaveTextContent("No download record");
});

it("a deleted runner shows no history", async () => {
  const drawer = await openDrawerFor({ ...dispute, runnerHandle: "a@b.com", runnerHistory: null });
  expect(within(drawer).getByText("This runner's other requests").nextSibling).toHaveTextContent("Runner deleted");
});

it("an unrefundable dispute says why in the drawer", async () => {
  const drawer = await openDrawerFor({
    ...dispute, refundable: false, refundBlockedReason: "no_payment", refundBlockedMessage: "No payment is on file.",
  });
  expect(within(drawer).getByText("Can it be refunded?").nextSibling).toHaveTextContent("No: No payment is on file.");
});

it("the drawer labels the order total as the order total", async () => {
  const drawer = await openDrawerFor({ ...dispute, itemChargeAmount: 40, orderSnapshot: { ...dispute.orderSnapshot, total: 330 } });
  expect(within(drawer).getByText("Order total")).toBeInTheDocument();
  expect(drawer).toHaveTextContent(/Order total.*330/);
  expect(within(drawer).queryByText("Total")).not.toBeInTheDocument();
});

// AX-57: drawer and detail values read as typed, wrap at 375, and link a no-handle photographer by id.
const notUpperCased = (els: HTMLElement[]) => { expect(els.length).toBeGreaterThan(0); for (const el of els) expect(el.closest(".uppercase")).toBeNull(); };
it("the drawer's subtitle, ids and dates are not upper-cased", async () => {
  fixtures.search = `row=${dispute.id}`;
  renderQueue();
  const drawer = await screen.findByRole("dialog");
  notUpperCased(within(drawer).getAllByText(/Reported Sep 10 · 2026/));
  notUpperCased(within(drawer).getAllByText("order-1"));
  notUpperCased(within(drawer).getAllByText(/^Sep 0?9 · 2026$/));
  // A title holds two handles, so it must wrap at 375.
  expect(within(drawer).getAllByRole("heading", { level: 2 })[0]).toHaveClass("break-words");
});

it("an empty Needs you list says no disputes are waiting", async () => {
  vi.mocked(api.get).mockImplementation(async (path) => {
    if (path.startsWith("/admin/disputes?")) return page([]);
    if (path === "/admin/kpis") return fixtures.kpis;
    throw new Error(`unexpected GET ${path}`);
  });
  seed([]);
  renderQueue();
  expect(await screen.findByText("No disputes waiting. You're caught up.")).toBeInTheDocument();
});

// FT-18 (C-36): one line under the view tabs says what each tab means.
it("a line under the view tabs says what each tab means", () => {
  seed([dispute]);
  renderQueue();
  expect(screen.getByText("Needs you: waiting for your decision. Overdue: past the due date on the dispute. Refund run: approved QR Ph refunds you still have to pay in PayMongo.")).toBeInTheDocument();
});

// FT-24 (C-39, C-60, B-40): a decided dispute leaves the drawer, and the toast says where it went.
describe("after a decision", () => {
  const denyReasons = { items: [{ code: "preference", label: "Doesn't like how it looks", runnerText: "The photo arrived as shown.", noteRequired: false }], total: 1 };
  // The server answers the decision with `after`, and its lists then hold `after`.
  function serve(after: Dispute) {
    vi.mocked(api.get).mockImplementation(async (path) => {
      if (path === "/admin/disputes/deny-reasons") return denyReasons;
      if (path.startsWith("/admin/disputes?")) return page([after]);
      if (path === "/admin/kpis") return fixtures.kpis;
      throw new Error(`unexpected GET ${path}`);
    });
    vi.spyOn(api, "post").mockResolvedValueOnce(after);
  }
  const lastToast = () => useToastStore.getState().toasts.at(-1);
  beforeEach(() => useToastStore.getState().clear());

  async function openDrawer(d: Dispute) {
    seed([d]);
    fixtures.search = `row=${d.id}`;
    renderQueue();
    return screen.findByRole("dialog");
  }

  async function approve(drawer: HTMLElement, user: ReturnType<typeof userEvent.setup>) {
    await user.click(within(drawer).getByRole("button", { name: "Approve refund…" }));
    await user.click(within(screen.getByRole("dialog", { name: `Approve refund · ${dispute.id}` })).getByRole("button", { name: "Review refund" }));
    await user.click(within(screen.getByRole("dialog", { name: "Confirm refund" })).getByRole("button", { name: "Confirm refund" }));
  }

  it("after Deny the drawer closes and the toast names the Closed tab", async () => {
    const user = userEvent.setup();
    serve({ ...dispute, status: "denied" });
    const drawer = await openDrawer(dispute);
    await user.click(within(drawer).getByRole("button", { name: "Deny…" }));
    const dialog = screen.getByRole("dialog", { name: `Deny ${dispute.id}` });
    await user.click(await within(dialog).findByRole("radio", { name: /Doesn't like/ }));
    await user.click(within(dialog).getByRole("button", { name: "Deny claim" }));
    await waitFor(() => expect(fixtures.router.replace).toHaveBeenCalledWith("/admin/disputes", { scroll: false }));
    expect(lastToast()).toMatchObject({ kind: "success", message: "Denied. The runner was sent your reason. It is now under Closed." });
  });

  it("after Escalate the drawer stays open and the toast says it stays under Needs you", async () => {
    const user = userEvent.setup();
    serve({ ...dispute, status: "escalated", attention: "escalated" });
    const drawer = await openDrawer(dispute);
    await user.click(within(drawer).getByRole("button", { name: "Escalate…" }));
    await user.click(within(screen.getByRole("dialog", { name: `Escalate ${dispute.id}` })).getByRole("button", { name: "Escalate" }));
    await waitFor(() => expect(lastToast()).toMatchObject({ kind: "success", message: "Escalated. It stays under Needs you." }));
    expect(fixtures.router.replace).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog")).toHaveTextContent(dispute.id);
  });

  it("after an approval that finished the refund the drawer closes", async () => {
    const user = userEvent.setup();
    serve({ ...dispute, status: "resolved", resolution: "refund_full", refundAmount: 100 });
    const drawer = await openDrawer(dispute);
    await approve(drawer, user);
    await waitFor(() => expect(fixtures.router.replace).toHaveBeenCalledWith("/admin/disputes", { scroll: false }));
    expect(lastToast()).toMatchObject({ kind: "success", message: "Refunded ₱100.00 · dispute-1" });
  });

  it("after a QR Ph approval the drawer stays open with the refund panel", async () => {
    const user = userEvent.setup();
    const qrph: Dispute = { ...dispute, orderSnapshot: { ...dispute.orderSnapshot, paymentMethod: "qrph" } };
    serve({ ...qrph, status: "escalated", resolution: "refund_full", refundAmount: 100, refundStatus: "manual_review" });
    const drawer = await openDrawer(qrph);
    await approve(drawer, user);
    await waitFor(() => expect(lastToast()).toMatchObject({ kind: "success", message: "Refund approved. Next, send the runner the refund link." }));
    expect(await within(screen.getByRole("dialog")).findByLabelText("Refund link")).toBeInTheDocument();
    expect(fixtures.router.replace).not.toHaveBeenCalled();
  });
});
