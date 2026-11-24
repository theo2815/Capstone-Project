import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { RemovalRequestList } from "./request-list";
import { draftCase, openCase, runnerCase } from "./removal-fixtures";

vi.mock("@/lib/api-photo-removal");
// The Waiting tab's count is the Inbox's removals count.
vi.mock("@/lib/admin-queues", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/admin-queues")>()),
  useAdminQueueCounts: () => ({ byKey: { removals: 3 }, total: 3 }),
}));
// A tiny in-memory URL so the list's URL-held filters behave as in the browser.
const nav = vi.hoisted(() => {
  let search = "";
  const listeners = new Set<() => void>();
  return {
    push: vi.fn(),
    get: () => search,
    set: (next: string) => { search = next; listeners.forEach((l) => l()); },
    subscribe: (l: () => void) => { listeners.add(l); return () => listeners.delete(l); },
  };
});
vi.mock("next/navigation", async () => {
  const { useSyncExternalStore } = await import("react");
  return {
    usePathname: () => "/admin/photo-removal-requests",
    useSearchParams: () => new URLSearchParams(useSyncExternalStore(nav.subscribe, nav.get)),
    useRouter: () => ({ replace: (url: string) => nav.set(url.split("?")[1] ?? ""), push: nav.push }),
  };
});
let client: QueryClient;
beforeEach(() => { useAuthStore.getState().startSession(admin); client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); });
afterEach(() => { nav.set(""); cleanup(); client.clear(); useAuthStore.getState().logout(); vi.resetAllMocks(); });

it("counts case photos from server phase counts because list rows carry no items", async () => {
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [{ ...draftCase, status: "OPEN", items: [],
    phaseCounts: { ...draftCase.phaseCounts, candidate: 9, verified: 2, excluded: 1, removalAuthorized: 1, removed: 1, retained: 0, retainedRestricted: 0 } }],
    total: 1, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(await screen.findByText(/4 photos/)).toBeInTheDocument();
});

it("the list shows a Runner badge and filters by source", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [
    { ...runnerCase, items: [] }, { ...openCase, id: "case-2", items: [] },
  ], total: 2, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  const [runnerRow, supportRow] = await screen.findAllByRole("link");
  expect(within(runnerRow).getByText("Runner")).toBeVisible();
  expect(within(runnerRow).queryByText("RUNNER-case-1")).toBeNull();
  expect(within(runnerRow).getByText("Open")).toBeVisible();
  expect(within(supportRow).getByText("Support")).toBeVisible();
  expect(within(supportRow).getByText("TICKET-42")).toBeVisible();

  await user.click(screen.getByLabelText("Source"));
  await user.click(screen.getByRole("option", { name: "Runner requests" }));
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, source: "RUNNER", needsAction: true }, expect.anything()));
  expect(nav.get()).toBe("source=RUNNER");
  await user.click(screen.getByRole("button", { name: "All" }));
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, source: "RUNNER" }, expect.anything()));
});

it("opens on the Waiting tab with its count, and each tab writes one filter", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  const tabs = within(screen.getByRole("group", { name: "Status" })).getAllByRole("button");
  expect(tabs.map((t) => t.textContent)).toEqual(["Waiting3", "Draft", "Open", "Closing", "Resolved", "Cancelled", "All"]);
  expect(tabs[0]).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText("Closing: a removal was authorized and the files are still being deleted.")).toBeVisible();
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, needsAction: true }, expect.anything()));
  expect(await screen.findByText("No requests waiting. You're caught up.")).toBeVisible();

  await user.click(screen.getByRole("button", { name: "Closing" }));
  expect(nav.get()).toBe("status=RESOLVING");
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, status: "RESOLVING" }, expect.anything()));
  expect(await screen.findByText("No requests match these filters.")).toBeVisible();

  await user.click(screen.getByRole("button", { name: "All" }));
  expect(nav.get()).toBe("needs=0");
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20 }, expect.anything()));
  expect(await screen.findByText("No removal requests yet.")).toBeVisible();

  await user.click(screen.getByRole("button", { name: /^Waiting/ }));
  expect(nav.get()).toBe("needs=1");
  expect(screen.getByRole("button", { name: /^Waiting/ })).toHaveAttribute("aria-pressed", "true");
});

it("shows skeleton rows while loading and a retry when the list fails", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.listAdminRemovalRequests).mockRejectedValueOnce(new Error("down")).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const view = render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(screen.getByRole("status", { name: "Loading removal requests" })).toHaveAttribute("aria-busy", "true");
  expect(view.container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);
  expect(await screen.findByText("Couldn't load removal requests.")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByText("No requests waiting. You're caught up.")).toBeVisible();
});

it("keeps filters in the URL across a remount and explains an empty filtered list", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const view = render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  await user.click(screen.getByLabelText("Source"));
  await user.click(screen.getByRole("option", { name: "Runner requests" }));
  expect(await screen.findByText("No requests match these filters.")).toBeVisible();
  expect(nav.get()).toBe("source=RUNNER");

  // Leaving for a case and pressing Back remounts the list on the same URL.
  view.unmount();
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(screen.getByLabelText("Source")).toHaveTextContent("Runner requests");

  await user.click(await screen.findByRole("button", { name: "Clear filters" }));
  expect(await screen.findByText("No requests waiting. You're caught up.")).toBeVisible();
  expect(nav.get()).toBe("");
});

it("ignores inherited object keys in the status and source filters", async () => {
  nav.set("status=constructor&source=toString");
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(await screen.findByText("No requests waiting. You're caught up.")).toBeVisible();
  expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, needsAction: true }, expect.anything());
});

// AX-39 (UA7): row values read as typed; mono caps stay for labels only.
it("a case row's date is in sentence case", async () => {
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [{ ...openCase, items: [] }], total: 1, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(await screen.findByText(/photos · Oct 08 · 2026/)).toBeInTheDocument();
});

// FU-14: below 768 px the event name takes the whole first line; count, date and pills wrap below it.
it("a case row keeps the event name on its own line on phones", async () => {
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [{ ...openCase, items: [] }], total: 1, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  const nameBlock = (await screen.findByText(openCase.event.name)).parentElement!;
  expect(nameBlock).toHaveClass("basis-full", "md:basis-0", "grow");
  expect(nameBlock.closest("a")).toHaveClass("flex-wrap");
});

it("the Inbox list says no requests are waiting", async () => {
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  nav.set("status=OPEN&needs=0");
  render(<QueryClientProvider client={client}><RemovalRequestList needsActionOnly /></QueryClientProvider>);
  expect(await screen.findByText("No requests waiting. You're caught up.")).toBeVisible();
  // The Inbox shows the waiting cases only: no status tabs, whatever the URL says.
  expect(screen.queryByRole("group", { name: "Status" })).toBeNull();
  expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, needsAction: true }, expect.anything());
});

it("FT-36: J then Enter opens the first request", async () => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [{ ...openCase, items: [] }], total: 1, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  const link = await screen.findByRole("link");
  act(() => { fireEvent.keyDown(document.body, { key: "j" }); });
  act(() => { fireEvent.keyDown(document.body, { key: "Enter" }); });
  expect(nav.push).toHaveBeenCalledWith(link.getAttribute("href"));
});
