import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Flag } from "@/lib/admin-flags";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { FlagsQueue } from "./flags-queue";

const navigation = vi.hoisted(() => ({ search: "", replace: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: navigation.replace }), usePathname: () => "/admin/flags", useSearchParams: () => new URLSearchParams(navigation.search) }));
vi.mock("@/lib/constants", async (original) => ({ ...await original<object>(), ADMIN_FLAGS_ENABLED: true }));
// Stubs, so the test can count how many modal instances the queue mounts.
vi.mock("./admin-flag-hide-modal", () => ({ AdminFlagHideModal: () => <div data-testid="hide-modal" /> }));
vi.mock("./admin-escalate-modal", () => ({ AdminEscalateModal: () => <div data-testid="escalate-modal" /> }));

const flag: Flag = { id: "flag-1", photoId: "photo-1", eventId: "event-1", eventName: "Race", photographerHandle: "photographer", reportedBy: "runner", reason: "low_quality", note: "Blurry", status: "open", reportedAt: "2026-09-10T00:00:00Z", reviewedAt: null, reviewedBy: null, reviewerNote: null, photoSnapshot: { alt: "Runner", kmMark: null, bib: null, thumbnailUrl: null }, targetKind: "photo", targetId: "photo-1", payoutHoldAt: null };
const relevanceFlag: Flag = { ...flag, id: "flag-2", photoId: null, reason: "batch_relevance_low", note: "event e1: copy 5/8", targetKind: "user", targetId: "user-1" };

// A small fake of GET /admin/flags: one status, q (id, handles, note), offset and limit.
function serve(flags: Flag[]) {
  vi.mocked(api.get).mockImplementation((async (url: string) => {
    if (url.includes("/relevance")) return { flagId: "flag-2", payoutHoldAt: null, batches: [] };
    const p = new URL(url, "http://test").searchParams;
    const status = p.get("status");
    const q = (p.get("q") ?? "").toLowerCase();
    const hits = flags.filter((f) => (!status || f.status === status)
      && (!q || [f.id, f.photographerHandle, f.reportedBy, f.note].some((v) => v.toLowerCase().includes(q))));
    const offset = Number(p.get("offset") ?? 0);
    const limit = Number(p.get("limit") ?? 200);
    return { items: hits.slice(offset, offset + limit), total: hits.length, offset, limit };
  }) as typeof api.get);
}
const closed = (n: number, status: Flag["status"], prefix = status) =>
  Array.from({ length: n }, (_, i): Flag => ({ ...flag, id: `${prefix}-${i}`, status, reviewedAt: "2026-09-11T00:00:00Z" }));
const flagUrls = () => vi.mocked(api.get).mock.calls.map(([url]) => String(url)).filter((url) => url.startsWith("/admin/flags?"));
const cardsIn = (slab: string) => within(document.getElementById(slab)!).queryAllByRole("listitem");

let client: QueryClient;
const mount = () => render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  navigation.search = ""; navigation.replace.mockClear();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  vi.spyOn(api, "get");
  serve([flag]);
});

it("each history slab reads its own statuses from the server", async () => {
  serve([flag, ...closed(2, "hidden"), ...closed(1, "dismissed"), ...closed(1, "resolved"), ...closed(1, "escalated")]);
  mount();
  await waitFor(() => expect(cardsIn("hidden")).toHaveLength(2));
  for (const status of ["hidden", "dismissed", "resolved", "escalated"]) {
    expect(flagUrls().some((url) => url.includes(`status=${status}&`))).toBe(true);
  }
  expect(cardsIn("dismissed")).toHaveLength(2);
  expect(document.getElementById("dismissed")).toHaveTextContent("2 flags");
});

it("Load more appends the next server page", async () => {
  serve(closed(60, "hidden"));
  mount();
  await waitFor(() => expect(cardsIn("hidden")).toHaveLength(50));
  expect(document.getElementById("hidden")).toHaveTextContent("60 flags");
  await userEvent.click(within(document.getElementById("hidden")!).getByRole("button", { name: "Load 10 more" }));
  await waitFor(() => expect(cardsIn("hidden")).toHaveLength(60));
  expect(flagUrls()).toContainEqual(expect.stringMatching(/status=hidden&offset=50&limit=50/));
});

it("History finds a flag past the newest 200, by photographer, through the server search", async () => {
  const old: Flag = { ...flag, id: "flag-old", status: "hidden", photographerHandle: "oldie" };
  serve([...closed(250, "hidden"), old]);
  navigation.search = "q=oldie";
  mount();
  await waitFor(() => expect(cardsIn("hidden")).toHaveLength(1));
  expect(within(document.getElementById("hidden")!).getByRole("listitem")).toHaveTextContent("oldie");
  expect(flagUrls()).toContainEqual(expect.stringMatching(/status=hidden&q=oldie&offset=0/));
  expect(screen.queryByText(/History shows the newest/)).not.toBeInTheDocument();
});

it("a ?row= link to a closed flag past the first page opens its drawer", async () => {
  serve(closed(60, "hidden"));
  navigation.search = "row=hidden-55";
  mount();
  expect(await screen.findByRole("dialog")).toHaveTextContent("hidden-55");
  expect(cardsIn("hidden")).toHaveLength(50);
  expect(navigation.replace).not.toHaveBeenCalled();
  expect(screen.queryByText("This flag is not in the open list.")).not.toBeInTheDocument();
});

it("one failed history list shows the alert, not an empty slab", async () => {
  const served = vi.mocked(api.get).getMockImplementation()!;
  vi.mocked(api.get).mockImplementation(((url: string, ...rest: never[]) =>
    url.includes("status=hidden") ? Promise.reject(new Error("offline")) : served(url, ...rest)) as typeof api.get);
  mount();
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load flags");
  expect(screen.queryByText("No hidden photos.")).not.toBeInTheDocument();
});

it("a new search starts each history slab at its first page", async () => {
  serve(closed(60, "hidden"));
  const view = mount();
  await waitFor(() => expect(cardsIn("hidden")).toHaveLength(50));
  await userEvent.click(within(document.getElementById("hidden")!).getByRole("button", { name: "Load 10 more" }));
  await waitFor(() => expect(cardsIn("hidden")).toHaveLength(60));
  // The search box writes ?q= after its 300 ms debounce; the lists follow the URL.
  navigation.search = "q=hidden-5";
  view.rerender(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
  await waitFor(() => expect(cardsIn("hidden")).toHaveLength(11));
  expect(flagUrls()).toContainEqual(expect.stringMatching(/status=hidden&q=hidden-5&offset=0&limit=50/));
  expect(flagUrls().filter((url) => url.includes("q=hidden-5&offset=50"))).toHaveLength(0);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("a shared ?row= link survives the first load and opens the drawer", async () => {
  navigation.search = "row=flag-1";
  const waiting: (() => void)[] = [];
  vi.mocked(api.get).mockImplementation(() => new Promise((yes) => { waiting.push(() => yes({ items: [flag], total: 1 })); }));
  const { container } = mount();
  expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();
  expect(screen.queryByText("No flags waiting. You're caught up.")).not.toBeInTheDocument();
  await act(async () => { waiting.forEach((release) => release()); });
  expect(await screen.findByRole("dialog")).toHaveTextContent("flag-1");
  expect(navigation.replace).not.toHaveBeenCalled();
});

it("a link to a flag that is not in the list says so and offers a search", async () => {
  navigation.search = "row=flag-404";
  mount();
  expect(await screen.findByText("This flag is not in the open list.")).toBeInTheDocument();
  await waitFor(() => expect(navigation.replace).toHaveBeenCalledWith("/admin/flags", { scroll: false }));
  await userEvent.click(screen.getByRole("button", { name: "Search all flags for flag-404" }));
  await waitFor(() => expect(navigation.replace).toHaveBeenLastCalledWith(expect.stringContaining("q=flag-404"), { scroll: false }));
});

it("a failed load shows an alert with Retry, never an empty queue", async () => {
  vi.mocked(api.get).mockRejectedValue(new Error("offline"));
  navigation.search = "row=flag-1";
  mount();
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Couldn't load flags");
  expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  expect(screen.queryByText("No flags waiting. You're caught up.")).not.toBeInTheDocument();
  expect(navigation.replace).not.toHaveBeenCalled();
});

it("a link opened over a stale cached list waits for the refetch", async () => {
  const fresh: Flag = { ...flag, id: "flag-new" };
  mount();
  await screen.findAllByRole("button", { name: "Hide…" });
  cleanup();
  await client.invalidateQueries({ queryKey: ["admin", "flags"] }, { cancelRefetch: false });
  serve([flag, fresh]);
  navigation.search = "row=flag-new";
  mount();
  expect(await screen.findByRole("dialog")).toHaveTextContent("flag-new");
  expect(screen.queryByText("This flag is not in the open list.")).not.toBeInTheDocument();
});

it("the missing-link notice goes when the admin searches", async () => {
  navigation.search = "row=flag-404";
  mount();
  await screen.findByText("This flag is not in the open list.");
  await userEvent.type(screen.getByRole("searchbox", { name: "Search flags" }), "x");
  expect(screen.queryByText("This flag is not in the open list.")).not.toBeInTheDocument();
});

it("drawer Hide and Hold payouts are danger buttons, not green", async () => {
  serve([flag, relevanceFlag]);
  navigation.search = "row=flag-1";
  mount();
  const hide = within(await screen.findByRole("dialog")).getByRole("button", { name: "Hide…" });
  expect(hide).toHaveClass("text-error", "min-h-[44px]");
  expect(hide).not.toHaveClass("bg-fresh");
  cleanup();
  navigation.search = "row=flag-2";
  mount();
  const hold = within(await screen.findByRole("dialog")).getByRole("button", { name: "Hold payouts" });
  expect(hold).toHaveClass("text-error", "min-h-[44px]");
  expect(hold).not.toHaveClass("bg-fresh");
});

it("fifty cards share one hide modal and one escalate modal", async () => {
  const many = Array.from({ length: 50 }, (_, i) => ({ ...flag, id: `flag-${i}`, reportedAt: `2026-09-${String(10 + (i % 18)).padStart(2, "0")}T00:00:00Z` }));
  serve(many);
  mount();
  await waitFor(() => expect(screen.getAllByRole("button", { name: "Hide…" })).toHaveLength(50));
  expect(screen.getAllByTestId("hide-modal")).toHaveLength(1);
  expect(screen.getAllByTestId("escalate-modal")).toHaveLength(1);
});

it("the page header shows no counts when the flags failed to load", async () => {
  vi.mocked(api.get).mockRejectedValue(new Error("offline"));
  const { default: AdminFlagsPage } = await import("@/app/admin/flags/page");
  render(<QueryClientProvider client={client}><AdminFlagsPage /></QueryClientProvider>);
  await screen.findByRole("alert");
  expect(screen.getByRole("banner")).not.toHaveTextContent(/0 open|0 hidden/);
});

it("a drawer whose flag leaves the lists closes without the missing-link notice", async () => {
  navigation.search = "row=flag-1";
  mount();
  expect(await screen.findByRole("dialog")).toHaveTextContent("flag-1");
  serve([]);
  await act(async () => { await client.invalidateQueries({ queryKey: ["admin", "flags"] }); });
  await waitFor(() => expect(navigation.replace).toHaveBeenCalledWith("/admin/flags", { scroll: false }));
  expect(screen.queryByText("This flag is not in the open list.")).not.toBeInTheDocument();
});

// AX-57: drawer and detail values read as typed, wrap at 375, and link a no-handle photographer by id.
const notUpperCased = (els: HTMLElement[]) => { expect(els.length).toBeGreaterThan(0); for (const el of els) expect(el.closest(".uppercase")).toBeNull(); };
it("the drawer's subtitle, handles and dates are not upper-cased", async () => {
  navigation.search = "row=flag-1";
  mount();
  const drawer = await screen.findByRole("dialog");
  notUpperCased(within(drawer).getAllByText(/Sep 10 · 2026/));
  notUpperCased(within(drawer).getAllByText(/@photographer/));
  for (const el of within(drawer).getAllByText("@photographer", { selector: "dd" })) expect(el).toHaveClass("break-words");
});

it("an empty open list says no flags are waiting", async () => {
  vi.mocked(api.get).mockResolvedValue({ items: [], total: 0 });
  mount();
  expect(await screen.findByText("No flags waiting. You're caught up.")).toBeInTheDocument();
  expect(screen.getByText("Waiting for a decision")).toBeInTheDocument();
});
