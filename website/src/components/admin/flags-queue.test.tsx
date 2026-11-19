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

let client: QueryClient;
const mount = () => render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  navigation.search = ""; navigation.replace.mockClear();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  vi.spyOn(api, "get").mockResolvedValue({ items: [flag], total: 1 });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("a shared ?row= link survives the first load and opens the drawer", async () => {
  navigation.search = "row=flag-1";
  const waiting: (() => void)[] = [];
  vi.mocked(api.get).mockImplementation(() => new Promise((yes) => { waiting.push(() => yes({ items: [flag], total: 1 })); }));
  const { container } = mount();
  expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();
  expect(screen.queryByText("Queue is clear — no open flags.")).not.toBeInTheDocument();
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
  expect(screen.queryByText("Queue is clear — no open flags.")).not.toBeInTheDocument();
  expect(navigation.replace).not.toHaveBeenCalled();
});

it("a link opened over a stale cached list waits for the refetch", async () => {
  const fresh: Flag = { ...flag, id: "flag-new" };
  mount();
  await screen.findAllByRole("button", { name: "Hide…" });
  cleanup();
  await client.invalidateQueries({ queryKey: ["admin", "flags"] }, { cancelRefetch: false });
  vi.mocked(api.get).mockResolvedValue({ items: [flag, fresh], total: 2 });
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
  vi.mocked(api.get).mockImplementation((async (url: string) =>
    url.includes("/relevance") ? { flagId: "flag-2", payoutHoldAt: null, batches: [] } : { items: [flag, relevanceFlag], total: 2 }) as typeof api.get);
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
  vi.mocked(api.get).mockResolvedValue({ items: many, total: 50 });
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
  vi.mocked(api.get).mockResolvedValue({ items: [], total: 0 });
  await act(async () => { await client.invalidateQueries({ queryKey: ["admin", "flags"] }); });
  await waitFor(() => expect(navigation.replace).toHaveBeenCalledWith("/admin/flags", { scroll: false }));
  expect(screen.queryByText("This flag is not in the open list.")).not.toBeInTheDocument();
});
