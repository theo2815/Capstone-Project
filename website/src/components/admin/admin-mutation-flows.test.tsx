import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Dispute } from "@/lib/admin-disputes";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import { useAdminDisputeStore } from "@/store/admin-dispute-store";
import { useAdminPayoutReportStore } from "@/store/admin-payout-report-store";
import { useToastStore } from "@/store/toast-store";
import { AdminDisputeActionAside } from "./admin-dispute-action-aside";
import { DisputesQueue } from "./disputes-queue";
import { PayoutReportsSection } from "./payout-reports-section";

const fixtures = vi.hoisted(() => ({
  disputes: [] as Dispute[],
  reports: [] as PayoutReport[],
  payouts: [],
  router: { replace: vi.fn() },
}));
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/inbox",
  useSearchParams: () => new URLSearchParams(),
}));
vi.mock("@/hooks/use-admin-data", () => ({
  useAdminDisputes: () => fixtures.disputes,
  useAdminPayoutReports: () => fixtures.reports,
  useAdminPayouts: () => fixtures.payouts,
}));

const dispute: Dispute = {
  id: "dispute-1", orderId: "order-1", photoId: "photo-1", eventId: "event-1",
  runnerHandle: "runner", photographerHandle: "photographer", reason: "low_quality",
  note: "The photo is blurry.", status: "open", reportedAt: "2026-09-10T00:00:00Z",
  resolvedAt: null, refundAmount: null, resolution: null,
  orderSnapshot: { total: 100, paymentMethod: "gcash", paidAt: "2026-09-09T00:00:00Z" },
  photoSnapshot: { alt: "Runner", bib: "123", kmMark: 5 },
};
const report: PayoutReport = {
  id: "report-1", payoutCycleId: "cycle-1", photographerId: "photographer-1",
  photographerName: "Photographer", handle: "photographer", reason: "not_received",
  note: "No funds received.", status: "open", reportedAt: "2026-09-10T00:00:00Z",
  acknowledgedAt: null, acknowledgeReply: null, resolvedAt: null, resolutionNote: null,
};

let client: QueryClient;
beforeEach(() => {
  // jsdom has no layout/scroll implementation.
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  fixtures.disputes = [dispute];
  fixtures.reports = [report];
  useAdminDisputeStore.getState().clear();
  useAdminPayoutReportStore.getState().clear();
  useToastStore.getState().clear();
  vi.spyOn(console, "error").mockImplementation(() => {});
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  cleanup();
  client.clear();
  Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView");
  vi.restoreAllMocks();
});

function pendingRequest(method: "post" | "fetch") {
  let reject!: (error: Error) => void;
  vi.spyOn(api, method).mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  return async () => { await act(async () => { reject(new Error("Backend unavailable")); }); };
}

describe.each(["detail", "queue"])("dispute %s", (surface) => {
  it.each(["refund", "resolve-deny", "deny", "escalate"])("keeps %s retryable after a failed request, then closes only on success", async (action) => {
    const user = userEvent.setup();
    render(surface === "detail"
      ? <AdminDisputeActionAside dispute={dispute} runnerHandle={dispute.runnerHandle} />
      : <DisputesQueue />);
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
    expect(useAdminDisputeStore.getState().overrides).toEqual({});
    await user.keyboard("{Escape}");
    expect(dialog).toBeInTheDocument();

    await fail();
    expect(dialog).toBeInTheDocument();
    expect(submit).toBeEnabled();
    expect(screen.getByRole("textbox")).toHaveValue(note);
    expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error"]);

    vi.mocked(api.post).mockResolvedValueOnce(dispute);
    await user.click(within(dialog).getByRole("button", { name: label }));
    expect(dialog).not.toBeInTheDocument();
    expect(useAdminDisputeStore.getState().log).toHaveLength(1);
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

  vi.mocked(api.fetch).mockResolvedValueOnce(report);
  await user.click(submit);
  expect(dialog).not.toBeInTheDocument();
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "payouts", "reports"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["photographer", "payouts", "reports"] });
  expect(useToastStore.getState().toasts.map((toast) => toast.kind)).toEqual(["error", "success"]);
});
