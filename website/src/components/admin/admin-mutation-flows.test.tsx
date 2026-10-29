import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Dispute } from "@/lib/admin-disputes";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import { useAuthStore } from "@/store/auth-store";
import { adminDisputeKeys, useDisputeActions } from "@/hooks/use-admin-disputes";
import { payoutReportKeys, usePayoutReportActions } from "@/hooks/use-payout-reports";
import { adminPayoutKeys } from "@/hooks/use-admin-payouts";
import { admin, dispute, report } from "@/test/fixtures/admin";
import { useToastStore } from "@/store/toast-store";
import { AdminDisputeActionAside } from "./admin-dispute-action-aside";
import { AdminResolveDisputeModal } from "./admin-resolve-dispute-modal";
import { DisputesQueue } from "./disputes-queue";
import { PayoutReportsSection } from "./payout-reports-section";

const fixtures = vi.hoisted(() => ({
  disputes: [] as Dispute[],
  reports: [] as PayoutReport[],
  payouts: [],
  search: "",
  router: { replace: vi.fn() },
}));

it.each(["dispute", "report"])("displays another view's pending %s action and releases after failure", async (family) => {
  const user = userEvent.setup();
  const wrapper = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const actions = renderHook(() => ({ disputes: useDisputeActions(), reports: usePayoutReportActions() }), { wrapper });
  render(family === "dispute" ? <AdminDisputeActionAside dispute={dispute} runnerHandle="runner" /> : <PayoutReportsSection />, { wrapper });
  if (family === "report") await user.click(screen.getByRole("button", { name: /report-1/i }));
  const fail = pendingRequest(family === "dispute" ? "post" : "fetch");
  let pending!: Promise<unknown>;
  act(() => { pending = family === "dispute" ? actions.result.current.disputes.deny(dispute.id, "Reason") : actions.result.current.reports.acknowledge(report.id, "Reply"); });
  await waitFor(() => expect(screen.getByRole("button", { name: "Resolve…" })).toBeDisabled());
  expect(screen.getByText(family === "dispute" ? "Denying…" : "Acknowledging…")).toBeInTheDocument();
  await fail();
  await pending;
  await waitFor(() => expect(screen.getByRole("button", { name: "Resolve…" })).toBeEnabled());
});
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/inbox",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

let client: QueryClient;
beforeEach(() => {
  // jsdom has no layout/scroll implementation.
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  fixtures.disputes = [dispute];
  fixtures.reports = [report];
  fixtures.search = "";
  fixtures.router.replace.mockClear();
  useAuthStore.getState().setUser(admin);
  vi.spyOn(api, "get").mockImplementation(async (path) => path.startsWith("/admin/disputes")
    ? { items: fixtures.disputes, total: fixtures.disputes.length }
    : path.startsWith("/admin/payouts/reports")
      ? { items: fixtures.reports, total: fixtures.reports.length }
      : { items: fixtures.payouts, total: fixtures.payouts.length });
  useToastStore.getState().clear();
  vi.spyOn(console, "error").mockImplementation(() => {});
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(adminDisputeKeys.list(admin.id), [dispute]);
  client.setQueryData(payoutReportKeys.adminList(admin.id), [report]);
  client.setQueryData(adminPayoutKeys.list(admin.id), []);
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  vi.restoreAllMocks();
});

function pendingRequest(method: "post" | "fetch") {
  let reject!: (error: Error) => void;
  vi.spyOn(api, method).mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  return async () => { await act(async () => { reject(new Error("Backend unavailable")); }); };
}

it("does not show an unknown refund outcome as refunded in the dispute drawer", async () => {
  const pending: Dispute & { refundStatus: string } = { ...dispute, status: "escalated", resolution: "refund_full", refundAmount: 100, refundStatus: "manual_review", activity: [] };
  fixtures.search = `row=${pending.id}`;
  client.setQueryData(adminDisputeKeys.list(admin.id), [pending]);
  render(<QueryClientProvider client={client}><DisputesQueue /></QueryClientProvider>);
  const drawer = await screen.findByRole("dialog");
  expect(drawer).toHaveTextContent("Escalated");
  expect(within(drawer).getByLabelText("Refund link")).toBeInTheDocument();
  expect(within(drawer).queryByText(/^refunded$/i)).not.toBeInTheDocument();
});

it.each(["requesting", "pending"])("shows no refund panel for an escalated refund still in flight (%s)", async (refundStatus) => {
  const inFlight: Dispute & { refundStatus: string } = { ...dispute, status: "escalated", resolution: "refund_full", refundAmount: 100, refundStatus };
  fixtures.search = `row=${inFlight.id}`;
  client.setQueryData(adminDisputeKeys.list(admin.id), [inFlight]);
  render(<QueryClientProvider client={client}><DisputesQueue /></QueryClientProvider>);
  const drawer = await screen.findByRole("dialog");
  expect(within(drawer).queryByLabelText("Refund link")).not.toBeInTheDocument();
});

it.each([false, true])("preserves the selected dispute while loading (stale cache: %s)", async (stale) => {
  fixtures.search = `row=${dispute.id}`;
  if (stale) client.setQueryData(adminDisputeKeys.list(admin.id), [], { updatedAt: 1 });
  else client.removeQueries({ queryKey: adminDisputeKeys.list(admin.id) });
  let resolve!: (value: unknown) => void;
  vi.mocked(api.get).mockReturnValueOnce(new Promise((yes) => { resolve = yes; }));
  render(<QueryClientProvider client={client}><DisputesQueue /></QueryClientProvider>);
  expect(fixtures.router.replace).not.toHaveBeenCalled();
  await act(async () => { resolve({ items: [dispute], total: 1 }); });
  expect(await screen.findByRole("dialog")).toHaveTextContent(dispute.id);
  expect(fixtures.router.replace).not.toHaveBeenCalled();
});

describe.each(["detail", "queue"])("dispute %s", (surface) => {
  it.each(["refund", "resolve-deny", "deny", "escalate"])("keeps %s retryable after a failed request, then closes only on success", async (action) => {
    const user = userEvent.setup();
    render(<QueryClientProvider client={client}>{surface === "detail"
      ? <AdminDisputeActionAside dispute={dispute} runnerHandle={dispute.runnerHandle} />
      : <DisputesQueue />}</QueryClientProvider>);
    if (surface === "queue") await user.click(screen.getByRole("button", { name: /dispute-1/i }));

    const fail = pendingRequest("post");
    const note = "Please review the supporting evidence.";
    let dialog: HTMLElement;
    let submit: HTMLElement;
    if (action === "refund" || action === "resolve-deny") {
      await user.click(screen.getByRole("button", { name: "Resolve…" }));
      dialog = screen.getByRole("dialog", { name: "Resolve dispute-1" });
      await user.type(within(dialog).getByRole("textbox"), note);
      if (action === "resolve-deny") {
        await user.click(within(dialog).getByRole("radio", { name: /Deny — no refund/ }));
        submit = within(dialog).getByRole("button", { name: "Deny" });
      } else {
        await user.click(within(dialog).getByRole("button", { name: "Continue" }));
        dialog = screen.getByRole("dialog", { name: "Confirm refund" });
        submit = within(dialog).getByRole("button", { name: "Confirm refund" });
      }
    } else {
      await user.click(screen.getByRole("button", { name: action === "deny" ? "Deny…" : "Escalate…" }));
      dialog = screen.getByRole("dialog", { name: `${action === "deny" ? "Deny" : "Escalate"} dispute-1` });
      await user.type(within(dialog).getByRole("textbox"), note);
      submit = within(dialog).getByRole("button", { name: action === "deny" ? "Deny claim" : "Escalate" });
    }
    const label = submit.textContent!;
    await user.click(submit);
    expect(submit).toBeDisabled();
    expect(submit).toHaveTextContent(action === "refund" || action === "resolve-deny" ? "Resolving" : action === "deny" ? "Denying" : "Escalating");
    await user.click(submit);
    expect(api.post).toHaveBeenCalledTimes(1);
    expect(useToastStore.getState().toasts).toEqual([]);
    expect(client.getQueryData(adminDisputeKeys.list(admin.id))).toEqual([dispute]);
    await user.keyboard("{Escape}");
    expect(dialog).toBeInTheDocument();

    await fail();
    expect(dialog).toBeInTheDocument();
    await waitFor(() => expect(submit).toBeEnabled());
    expect(screen.getByRole("textbox")).toHaveValue(note);
    expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error"]);

    const confirmed: Dispute = {
      ...dispute, status: action === "escalate" ? "escalated" : action === "deny" ? "denied" : "resolved",
      activity: [{ id: "server-decision", decidedAt: "2026-09-10T01:00:00Z", decision: "resolved", resolution: "deny", refundAmount: null, reason: note }],
    };
    fixtures.disputes = [confirmed];
    vi.mocked(api.post).mockResolvedValueOnce(confirmed);
    await user.click(within(dialog).getByRole("button", { name: label }));
    expect(dialog).not.toBeInTheDocument();
    expect(client.getQueryData(adminDisputeKeys.detail(admin.id, dispute.id))).toEqual(confirmed);
    expect(useToastStore.getState().toasts).toHaveLength(2);
    expect(useToastStore.getState().toasts[1].kind).not.toBe("error");
  });
});

it.each([
  ["detail", "open"], ["detail", "escalated"],
  ["queue", "open"], ["queue", "escalated"],
] as const)(
  "uses the disputed item charge on %s, and reports %s as pending", async (surface, status) => {
  const user = userEvent.setup();
  const priced: Dispute = {
    ...dispute, itemChargeAmount: 127.5,
    orderSnapshot: { ...dispute.orderSnapshot, total: 300 },
  };
  fixtures.disputes = [priced];
  client.setQueryData(adminDisputeKeys.list(admin.id), [priced]);
  render(<QueryClientProvider client={client}>{surface === "detail"
    ? <AdminDisputeActionAside dispute={priced} runnerHandle={priced.runnerHandle} />
    : <DisputesQueue />}</QueryClientProvider>);
  if (surface === "queue") await user.click(screen.getByRole("button", { name: /dispute-1/i }));
  await user.click(screen.getByRole("button", { name: "Resolve…" }));
  const form = screen.getByRole("dialog", { name: "Resolve dispute-1" });
  expect(within(form).getByText(/Full refund/)).toHaveTextContent("₱127.50");
  await user.type(within(form).getByRole("textbox"), "Refund approved.");
  await user.click(within(form).getByRole("button", { name: "Continue" }));
  const confirm = screen.getByRole("dialog", { name: "Confirm refund" });
  expect(confirm).toHaveTextContent("₱127.50");

  const pending: Dispute = { ...priced, status, resolution: "refund_full", refundAmount: 127.5 };
  vi.spyOn(api, "post").mockResolvedValueOnce(pending);
  await user.click(within(confirm).getByRole("button", { name: "Confirm refund" }));
  expect(api.post).toHaveBeenCalledWith(`/admin/disputes/${dispute.id}/resolve`, {
    resolution: "refund_full", refundAmount: 127.5, reason: "Refund approved.",
  });
  expect(useToastStore.getState().toasts.at(-1)?.message).toMatch(/Refund requested|Refund pending/);
  expect(useToastStore.getState().toasts.at(-1)?.message).not.toMatch(/Refunded/);
  },
);

it.each(["detail", "queue"])("labels a provider-confirmed %s refund with the returned amount", async (surface) => {
  const user = userEvent.setup();
  render(<QueryClientProvider client={client}>{surface === "detail"
    ? <AdminDisputeActionAside dispute={dispute} runnerHandle={dispute.runnerHandle} />
    : <DisputesQueue />}</QueryClientProvider>);
  if (surface === "queue") await user.click(screen.getByRole("button", { name: /dispute-1/i }));
  await user.click(screen.getByRole("button", { name: "Resolve…" }));
  const form = screen.getByRole("dialog", { name: "Resolve dispute-1" });
  await user.type(within(form).getByRole("textbox"), "Refund approved.");
  await user.click(within(form).getByRole("button", { name: "Continue" }));
  const confirm = screen.getByRole("dialog", { name: "Confirm refund" });
  const returned: Dispute = { ...dispute, status: "resolved", resolution: "refund_full", refundAmount: 85 };
  vi.spyOn(api, "post").mockResolvedValueOnce(returned);
  await user.click(within(confirm).getByRole("button", { name: "Confirm refund" }));
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("Refunded");
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("₱85.00");
});

it.each([null, 0, undefined])("disables monetary resolve when the item charge is %s", async (itemChargeAmount) => {
  const user = userEvent.setup();
  render(<AdminResolveDisputeModal open onClose={() => {}} onContinue={() => {}}
    disputeId="dispute-1" orderTotal={300} itemChargeAmount={itemChargeAmount} />);
  const dialog = screen.getByRole("dialog", { name: "Resolve dispute-1" });
  await user.type(within(dialog).getByRole("textbox"), "Reviewed the request.");
  if (itemChargeAmount === 0) expect(dialog).toHaveTextContent("This item has no refundable charge.");
  expect(within(dialog).getByRole("button", { name: "Continue" })).toBeDisabled();
  await user.click(within(dialog).getByRole("radio", { name: /Deny/ }));
  expect(within(dialog).getByRole("button", { name: "Deny" })).toBeEnabled();
});

it("accepts at most 500 resolve-note characters and only centavo partial amounts below the item charge", async () => {
  const user = userEvent.setup();
  const onContinue = vi.fn();
  render(<AdminResolveDisputeModal open onClose={() => {}} onContinue={onContinue}
    disputeId="dispute-1" orderTotal={300} itemChargeAmount={127.5} />);
  const dialog = screen.getByRole("dialog", { name: "Resolve dispute-1" });
  const note = within(dialog).getByRole("textbox");
  fireEvent.change(note, { target: { value: "a".repeat(501) } });
  expect(note).toHaveValue("a".repeat(500));
  await user.click(within(dialog).getByRole("radio", { name: /Partial refund/ }));
  const amount = within(dialog).getByRole("spinbutton");
  for (const invalid of ["0", "127.5", "127.501", "1.001"]) {
    fireEvent.change(amount, { target: { value: invalid } });
    expect(within(dialog).getByRole("button", { name: "Continue" })).toBeDisabled();
  }
  fireEvent.change(amount, { target: { value: "127.49" } });
  await user.click(within(dialog).getByRole("button", { name: "Continue" }));
  expect(onContinue).toHaveBeenCalledWith({ resolution: "refund_partial", refundAmount: 127.49, note: "a".repeat(500) });
});

it.each(["acknowledge", "resolve"])("keeps payout report %s open on failure and refreshes only on success", async (action) => {
  const user = userEvent.setup();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  render(<QueryClientProvider client={client}><PayoutReportsSection /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: /report-1/i }));
  await user.click(screen.getByRole("button", { name: action === "acknowledge" ? "Acknowledge…" : "Resolve…" }));
  const dialog = screen.getByRole("dialog", { name: new RegExp(`${action} · report-1`, "i") });
  const note = "We are investigating the transfer.";
  await user.type(within(dialog).getByRole("textbox"), note);
  const submit = within(dialog).getByRole("button", { name: action === "acknowledge" ? "Send acknowledgement" : "Resolve report" });
  const fail = pendingRequest("fetch");

  await user.click(submit);
  expect(submit).toBeDisabled();
  expect(submit).toHaveTextContent(action === "acknowledge" ? "Acknowledging" : "Resolving");
  expect(invalidate).not.toHaveBeenCalled();
  expect(useToastStore.getState().toasts).toEqual([]);
  await fail();
  expect(dialog).toBeInTheDocument();
  await waitFor(() => expect(submit).toBeEnabled());
  expect(within(dialog).getByRole("textbox")).toHaveValue(note);
  expect(invalidate).not.toHaveBeenCalled();
  expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error"]);

  const confirmed: PayoutReport = { ...report, status: action === "acknowledge" ? "acknowledged" : "resolved" };
  fixtures.reports = [confirmed];
  vi.mocked(api.fetch).mockResolvedValueOnce(confirmed);
  await user.click(submit);
  expect(dialog).not.toBeInTheDocument();
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "payouts", "reports"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["photographer", "payouts", "reports"] });
  expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error", "success"]);
});

it.each(["detail", "queue"])("offers no Escalate on a denied dispute in the %s view", async (surface) => {
  const user = userEvent.setup();
  const denied: Dispute = { ...dispute, status: "denied" };
  fixtures.disputes = [denied];
  client.setQueryData(adminDisputeKeys.list(admin.id), [denied]);
  render(<QueryClientProvider client={client}>{surface === "detail"
    ? <AdminDisputeActionAside dispute={denied} runnerHandle={denied.runnerHandle} />
    : <DisputesQueue />}</QueryClientProvider>);
  if (surface === "queue") await user.click(screen.getByRole("button", { name: /dispute-1/i }));
  // The server escalates open disputes only.
  expect(screen.queryByRole("button", { name: "Escalate…" })).not.toBeInTheDocument();
  expect(screen.getByText("This dispute is closed. No further actions are available.")).toBeInTheDocument();
});
