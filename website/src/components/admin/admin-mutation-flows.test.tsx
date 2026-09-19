import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Dispute } from "@/lib/admin-disputes";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import { useAuthStore } from "@/store/auth-store";
import { adminDisputeKeys } from "@/hooks/use-admin-disputes";
import { payoutReportKeys } from "@/hooks/use-payout-reports";
import { adminPayoutKeys } from "@/hooks/use-admin-payouts";
import { admin, dispute, report } from "@/test/fixtures/admin";
import { useToastStore } from "@/store/toast-store";
import { AdminDisputeActionAside } from "./admin-dispute-action-aside";
import { DisputesQueue } from "./disputes-queue";
import { PayoutReportsSection } from "./payout-reports-section";

const fixtures = vi.hoisted(() => ({
  disputes: [] as Dispute[],
  reports: [] as PayoutReport[],
  payouts: [],
  search: "",
  router: { replace: vi.fn() },
}));
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
    await user.click(submit);
    expect(api.post).toHaveBeenCalledTimes(1);
    expect(useToastStore.getState().toasts).toEqual([]);
    expect(client.getQueryData(adminDisputeKeys.list(admin.id))).toEqual([dispute]);
    await user.keyboard("{Escape}");
    expect(dialog).toBeInTheDocument();

    await fail();
    expect(dialog).toBeInTheDocument();
    expect(submit).toBeEnabled();
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
  expect(invalidate).not.toHaveBeenCalled();
  expect(useToastStore.getState().toasts).toEqual([]);
  await fail();
  expect(dialog).toBeInTheDocument();
  expect(submit).toBeEnabled();
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
