import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as removalApi from "@/lib/api-photo-removal";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { RemovalRequestList } from "./request-list";
import { draftCase, openCase, runnerCase } from "./removal-fixtures";

vi.mock("@/lib/api-photo-removal");
// A tiny in-memory URL so the list's URL-held filters behave as in the browser.
const nav = vi.hoisted(() => {
  let search = "";
  const listeners = new Set<() => void>();
  return {
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
    useRouter: () => ({ replace: (url: string) => nav.set(url.split("?")[1] ?? ""), push: vi.fn() }),
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

  await user.selectOptions(screen.getByLabelText("Source"), "RUNNER");
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, source: "RUNNER" }, expect.anything()));
  await user.click(screen.getByLabelText("Needs action"));
  await waitFor(() => expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20, source: "RUNNER", needsAction: true }, expect.anything()));
});

it("keeps filters in the URL across a remount and explains an empty filtered list", async () => {
  const user = userEvent.setup();
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  const view = render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  await user.selectOptions(screen.getByLabelText("Source"), "RUNNER");
  expect(await screen.findByText("No cases match these filters.")).toBeVisible();
  expect(nav.get()).toBe("source=RUNNER");

  // Leaving for a case and pressing Back remounts the list on the same URL.
  view.unmount();
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(screen.getByLabelText("Source")).toHaveValue("RUNNER");

  await user.click(await screen.findByRole("button", { name: "Clear filters" }));
  expect(await screen.findByText("No removal cases yet.")).toBeVisible();
  expect(nav.get()).toBe("");
});

it("ignores inherited object keys in the status and source filters", async () => {
  nav.set("status=constructor&source=toString");
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(await screen.findByText("No removal cases yet.")).toBeVisible();
  expect(removalApi.listAdminRemovalRequests).toHaveBeenLastCalledWith({ offset: 0, limit: 20 }, expect.anything());
});

// AX-39 (UA7): row values read as typed; mono caps stay for labels only.
it("a case row's date is in sentence case", async () => {
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [{ ...openCase, items: [] }], total: 1, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList /></QueryClientProvider>);
  expect(await screen.findByText(/photos · Oct 08 · 2026/)).toBeInTheDocument();
});

it("the Inbox list says no requests are waiting", async () => {
  vi.mocked(removalApi.listAdminRemovalRequests).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 });
  render(<QueryClientProvider client={client}><RemovalRequestList needsActionOnly /></QueryClientProvider>);
  expect(await screen.findByText("No requests waiting. You're caught up.")).toBeVisible();
});
