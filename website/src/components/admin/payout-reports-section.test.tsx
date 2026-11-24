import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { admin, payoutCycle, report } from "@/test/fixtures/admin";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";
import { PayoutReportsSection } from "./payout-reports-section";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/admin/payouts",
  useSearchParams: () => new URLSearchParams(),
}));

let client: QueryClient;
let reports: PayoutReport[];
let total: number;
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  reports = [report];
  total = 1;
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockImplementation(async (path: string) => {
    if (path === "/admin/kpis") return { openPayoutReports: 3 };
    if (path.startsWith("/admin/payouts/reports")) {
      const offset = Number(new URLSearchParams(path.split("?")[1]).get("offset"));
      return { items: reports, total, offset, limit: 50 };
    }
    return { items: [], total: 0, offset: 0, limit: 200 };
  });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
});

function renderSection() {
  return render(<QueryClientProvider client={client}><PayoutReportsSection /></QueryClientProvider>);
}
const reportCalls = () => vi.mocked(api.get).mock.calls.map(([path]) => path).filter((p) => p.startsWith("/admin/payouts/reports"));

it("sends the status filter to the server", async () => {
  const user = userEvent.setup();
  renderSection();
  await waitFor(() => expect(reportCalls().some((p) => p.includes("status=open"))).toBe(true));
  await user.click(screen.getByRole("button", { name: /^Resolved/ }));
  await waitFor(() => expect(reportCalls().some((p) => p.includes("status=resolved"))).toBe(true));
});

it("shows the open count from the KPIs in the header", async () => {
  renderSection();
  expect(await screen.findByText("3 open")).toBeInTheDocument();
});

it("links the cycle id to its payout", async () => {
  renderSection();
  expect(await screen.findByRole("link", { name: "cycle-1" })).toHaveAttribute("href", "/admin/payouts?row=cycle-1");
});

it("makes the card's title its only button", async () => {
  renderSection();
  const item = (await screen.findByText("Photographer", { selector: "button" })).closest("li")!;
  expect(within(item).getAllByRole("button")).toHaveLength(1);
});

it("loads the next page from the server", async () => {
  const user = userEvent.setup();
  total = 2;
  renderSection();
  await screen.findByText("Photographer", { selector: "button" });
  reports = [{ ...report, id: "report-2", photographerName: "Second" }];
  await user.click(screen.getByRole("button", { name: "Load 1 more" }));
  expect(await screen.findByText("Second", { selector: "button" })).toBeInTheDocument();
  expect(reportCalls().some((p) => p.includes("offset=1"))).toBe(true);
});

it("is the payout-reports anchor the rail links to", async () => {
  const { container } = renderSection();
  await screen.findByText("Photographer", { selector: "button" });
  expect(container.querySelector("section#payout-reports")).not.toBeNull();
});

it("names each card's button with its report id", async () => {
  reports = [report, { ...report, id: "report-2" }];
  total = 2;
  renderSection();
  expect(await screen.findByRole("button", { name: /report-1/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /report-2/ })).toBeInTheDocument();
});

it("keeps a typed reply when a refresh drops the report from the list", async () => {
  const user = userEvent.setup();
  renderSection();
  await user.click(await screen.findByRole("button", { name: /report-1/ }));
  await user.click(screen.getByRole("button", { name: "Acknowledge…" }));
  const dialog = screen.getByRole("dialog", { name: /acknowledge · report-1/i });
  await user.type(within(dialog).getByRole("textbox"), "Checking with GCash.");
  reports = [];
  total = 0;
  await client.invalidateQueries({ queryKey: ["admin", "payouts", "reports"] });
  await waitFor(() => expect(screen.getByText("No open reports. You're caught up.")).toBeInTheDocument());
  // Query again: the node captured above would keep its value even after unmounting.
  expect(within(screen.getByRole("dialog", { name: /acknowledge · report-1/i })).getByRole("textbox")).toHaveValue("Checking with GCash.");
});

it("FT-41: a report whose payout is outside the list shows its amount, status, paid date and reference in the drawer", async () => {
  const user = userEvent.setup();
  const paid = { ...payoutCycle, status: "paid" as const, paidAt: "2026-09-12T00:00:00Z", paymentReference: "GC-12345" };
  const fallback = vi.mocked(api.get).getMockImplementation()!;
  // The 200-row list holds no cycle-1; only a lookup by its id finds it.
  vi.mocked(api.get).mockImplementation(async (path: string, ...rest) =>
    path.startsWith("/admin/payouts?") && path.includes("q=cycle-1") ? { items: [paid], total: 1, offset: 0, limit: 200 } : fallback(path, ...rest));
  const byId = () => vi.mocked(api.get).mock.calls.filter(([p]) => p.includes("q=cycle-1")).length;
  renderSection();
  const card = (await screen.findByRole("button", { name: /report-1/ })).closest("article")!;
  // No request per card: the list card shows no amount it does not already have.
  expect(card).not.toHaveTextContent(formatPrice(paid.amount));
  expect(byId()).toBe(0);
  await user.click(within(card).getByRole("button", { name: /report-1/ }));
  const drawer = screen.getByRole("dialog");
  await waitFor(() => expect(drawer).toHaveTextContent(`Amount · ${formatPrice(paid.amount)}`));
  expect(drawer).toHaveTextContent("Status · Paid");
  expect(drawer).toHaveTextContent(`Paid on · ${formatLongDate(paid.paidAt)}`);
  expect(drawer).toHaveTextContent("Reference · GC-12345");
  expect(byId()).toBe(1);
});

it("FT-41: a pending payout shows its status, and no paid date or reference", async () => {
  const user = userEvent.setup();
  const fallback = vi.mocked(api.get).getMockImplementation()!;
  vi.mocked(api.get).mockImplementation(async (path: string, ...rest) =>
    path.includes("q=cycle-1") ? { items: [payoutCycle], total: 1, offset: 0, limit: 200 } : fallback(path, ...rest));
  renderSection();
  await user.click(await screen.findByRole("button", { name: /report-1/ }));
  const drawer = screen.getByRole("dialog");
  await waitFor(() => expect(drawer).toHaveTextContent("Status · Pending review"));
  expect(drawer).not.toHaveTextContent("Paid on");
  expect(drawer).not.toHaveTextContent("Reference");
});

it("says what each report tab means", async () => {
  renderSection();
  expect(await screen.findByText("Open: not answered yet. Acknowledged: you replied and are still working on it. Resolved: closed.")).toBeInTheDocument();
});

it("after Acknowledge, the toast names the tab the report moved to", async () => {
  const user = userEvent.setup();
  vi.spyOn(api, "fetch").mockResolvedValue({ ...report, status: "acknowledged" });
  renderSection();
  await user.click(await screen.findByRole("button", { name: /report-1/ }));
  await user.click(screen.getByRole("button", { name: "Acknowledge…" }));
  const dialog = screen.getByRole("dialog", { name: /acknowledge · report-1/i });
  await user.type(within(dialog).getByRole("textbox"), "Checking with GCash.");
  await user.click(within(dialog).getByRole("button", { name: "Send acknowledgement" }));
  await waitFor(() => expect(useToastStore.getState().toasts.at(-1)?.message).toBe("Reply sent. The report is now under Acknowledged."));
});

it("the Resolve dialog says it is final, and its toast says the note was sent", async () => {
  const user = userEvent.setup();
  vi.spyOn(api, "fetch").mockResolvedValue({ ...report, status: "resolved" });
  renderSection();
  await user.click(await screen.findByRole("button", { name: /report-1/ }));
  await user.click(screen.getByRole("button", { name: "Resolve…" }));
  const dialog = screen.getByRole("dialog", { name: /resolve · report-1/i });
  expect(within(dialog).getByText("Closes the report and sends your note to @photographer. Say what happened and how it was fixed, with the transfer reference. This is final: a resolved report cannot be reopened.")).toBeInTheDocument();
  await user.type(within(dialog).getByRole("textbox"), "Sent again, ref 12345.");
  await user.click(within(dialog).getByRole("button", { name: "Resolve report" }));
  await waitFor(() => expect(useToastStore.getState().toasts.at(-1)?.message).toBe("Report resolved. The photographer was sent your note."));
});

it.each(["Acknowledge", "Resolve"])("the %s… dialog says the note needs 10 characters", async (verb) => {
  const user = userEvent.setup();
  renderSection();
  await user.click(await screen.findByRole("button", { name: /report-1/ }));
  await user.click(screen.getByRole("button", { name: `${verb}…` }));
  const textbox = within(screen.getByRole("dialog", { name: new RegExp(`${verb} · report-1`, "i") })).getByRole("textbox");
  expect(textbox).toHaveAccessibleDescription("At least 10 characters.");
  expect(textbox).not.toHaveAttribute("placeholder", expect.stringMatching(/24h|MY-32C887R/));
});

it("loading shows skeleton rows, and a failed load says Couldn't load payout reports. with Retry", async () => {
  vi.mocked(api.get).mockImplementation(() => new Promise(() => {}));
  const { container, unmount } = renderSection();
  expect(container.querySelector('[aria-busy="true"] .animate-pulse')).not.toBeNull();
  expect(screen.queryByText(/Loading/)).toBeNull();
  unmount();
  client.clear();
  vi.mocked(api.get).mockRejectedValue(new Error("down"));
  renderSection();
  const alert = await screen.findByRole("alert");
  expect(within(alert).getByText("Couldn't load payout reports.")).toBeInTheDocument();
  expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
});

it("FT-36: J then Enter opens the first report", async () => {
  renderSection();
  await screen.findByRole("button", { name: /report-1/ });
  act(() => { fireEvent.keyDown(document.body, { key: "j" }); });
  act(() => { fireEvent.keyDown(document.body, { key: "Enter" }); });
  expect(await screen.findByRole("dialog")).toBeInTheDocument();
});
