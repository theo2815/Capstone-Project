import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { RelevanceAlert } from "@/lib/admin-relevance";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { AdminInboxFilterChips } from "@/components/admin/admin-inbox-filter-chips";
import { RelevanceAlertsInbox } from "./relevance-alerts-inbox";

const alert = (id: string, extra: Partial<RelevanceAlert> = {}): RelevanceAlert => ({
  alertId: id, caseId: "c1", eventId: "e1", eventName: "Cebu City Marathon 2003", photographerHandle: "juanshots",
  photographerName: "Juan Cruz", kind: "BIG_HIDE", wouldHave: true, photos: 640, createdAt: "2003-02-14T03:00:00Z", seenAt: null,
  ...extra,
});

let unseen: RelevanceAlert[];
let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  unseen = [alert("a1"), alert("a2", { kind: "COPIED", wouldHave: false, photos: null })];
  vi.spyOn(api, "get").mockImplementation((async (url: string) => {
    if (!url.startsWith("/admin/relevance/alerts?unseen=true")) throw new Error(`unexpected GET ${url}`);
    return { items: unseen, total: unseen.length, offset: 0, limit: 50 };
  }) as typeof api.get);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
const wrap = (ui: React.ReactNode) => render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);

it("the inbox chip counts unseen alerts", async () => {
  wrap(<AdminInboxFilterChips value="verifications" onChange={() => {}} />);
  const chip = await screen.findByRole("tab", { name: "Relevance alerts 2 unseen" });
  expect(chip).toHaveAttribute("aria-selected", "false");
  // The other chips carry no count, as before.
  expect(screen.getByRole("tab", { name: "Disputes" })).toBeInTheDocument();
});

it("marking an alert seen waits for the server", async () => {
  const user = userEvent.setup();
  let release!: (a: RelevanceAlert) => void;
  const post = vi.spyOn(api, "post").mockImplementation(() => new Promise((r) => { release = r as typeof release; }));
  wrap(<RelevanceAlertsInbox />);
  const first = await screen.findByRole("listitem", { name: /Big hide/ });
  expect(within(first).getByText("Would have hidden")).toBeInTheDocument();
  expect(within(first).getByRole("link", { name: /Open case/ })).toHaveAttribute("href", "/admin/relevance?tab=all&row=c1");
  const second = screen.getByRole("listitem", { name: /Copied photos/ });
  expect(within(second).queryByText("Would have hidden")).not.toBeInTheDocument();
  // Each Mark seen names its own alert, so a screen reader can tell the rows apart.
  expect(within(second).getByRole("button", { name: "Mark seen: Copied photos · Cebu City Marathon 2003" })).toBeInTheDocument();

  await user.click(within(first).getByRole("button", { name: "Mark seen: Big hide · Cebu City Marathon 2003" }));
  expect(post).toHaveBeenCalledWith("/admin/relevance/alerts/a1/seen");
  // Nothing moves until the server answers.
  expect(within(first).getByRole("button", { name: "Mark seen: Big hide · Cebu City Marathon 2003" })).toBeDisabled();
  expect(screen.getAllByRole("listitem")).toHaveLength(2);
  unseen = [unseen[1]!];
  release(alert("a1", { seenAt: "2003-02-14T04:00:00Z" }));
  await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(1));
  expect(screen.queryByRole("listitem", { name: /Big hide/ })).not.toBeInTheDocument();
});

it("a failed mark keeps the alert and says why", async () => {
  const user = userEvent.setup();
  vi.spyOn(api, "post").mockRejectedValue(new Error("down"));
  wrap(<RelevanceAlertsInbox />);
  const first = await screen.findByRole("listitem", { name: /Big hide/ });
  await user.click(within(first).getByRole("button", { name: "Mark seen: Big hide · Cebu City Marathon 2003" }));
  await waitFor(() => expect(useToastStore.getState().toasts.some((t) => t.kind === "error")).toBe(true));
  expect(within(first).getByRole("button", { name: "Mark seen: Big hide · Cebu City Marathon 2003" })).toBeEnabled();
  expect(screen.getAllByRole("listitem")).toHaveLength(2);
});

it("no unseen alerts says so", async () => {
  unseen = [];
  wrap(<RelevanceAlertsInbox />);
  expect(await screen.findByText("No unseen relevance alerts.")).toBeInTheDocument();
});
