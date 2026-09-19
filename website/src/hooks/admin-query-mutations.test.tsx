import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { fetchAdminDispute } from "@/lib/api-admin";
import type { Dispute } from "@/lib/admin-disputes";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import type { PaginatedResponse } from "@/types/api";
import { admin, dispute, report } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { adminDisputeKeys, useAdminDisputes, useDisputeActions } from "./use-admin-disputes";
import { payoutReportKeys, useAdminPayoutReports, usePhotographerPayoutReports, usePayoutReportActions } from "./use-payout-reports";

const reason = "Reviewed the evidence.";
const resolution = { resolution: "refund_partial" as const, refundAmount: 25, reason };
const confirmedDispute: Dispute = {
  ...dispute, status: "resolved", resolution: "refund_partial", refundAmount: 25,
  resolvedAt: "2026-09-10T01:02:03Z",
  activity: [{ id: "server-log-1", decision: "resolved", resolution: "refund_partial", refundAmount: 25, reason, decidedAt: "2026-09-10T01:02:03Z" }],
};
const confirmedReport: PayoutReport = {
  ...report, status: "resolved", resolutionNote: reason, resolvedAt: "2026-09-10T01:02:03Z",
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function useActionRunner(action: string) {
  const disputes = useDisputeActions();
  const reports = usePayoutReportActions();
  return () => {
    switch (action) {
      case "resolve dispute": return disputes.resolve(dispute.id, resolution);
      case "deny dispute": return disputes.deny(dispute.id, reason);
      case "escalate dispute": return disputes.escalate(dispute.id, reason);
      case "acknowledge report": return reports.acknowledge(report.id, reason);
      default: return reports.resolve(report.id, reason);
    }
  };
}

const cases = [
  { action: "resolve dispute", method: "post" as const, endpoint: `/admin/disputes/${dispute.id}/resolve`, payload: resolution, base: dispute, confirmed: confirmedDispute, domain: "disputes" },
  { action: "deny dispute", method: "post" as const, endpoint: `/admin/disputes/${dispute.id}/deny`, payload: { reason }, base: dispute, confirmed: { ...confirmedDispute, status: "denied" as const, resolution: "deny" as const, refundAmount: null }, domain: "disputes" },
  { action: "escalate dispute", method: "post" as const, endpoint: `/admin/disputes/${dispute.id}/escalate`, payload: { reason }, base: dispute, confirmed: { ...dispute, status: "escalated" as const, activity: [{ ...confirmedDispute.activity![0], decision: "escalated" as const }] }, domain: "disputes" },
  { action: "acknowledge report", method: "fetch" as const, endpoint: `/admin/payouts/reports/${report.id}/acknowledge`, payload: { method: "PATCH", body: JSON.stringify({ reply: reason }) }, base: report, confirmed: { ...report, status: "acknowledged" as const, acknowledgeReply: reason, acknowledgedAt: "2026-09-10T01:02:03Z" }, domain: "reports" },
  { action: "resolve report", method: "fetch" as const, endpoint: `/admin/payouts/reports/${report.id}/resolve`, payload: { method: "PATCH", body: JSON.stringify({ resolutionNote: reason }) }, base: report, confirmed: confirmedReport, domain: "reports" },
];

let client: QueryClient;
function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  vi.spyOn(console, "error").mockImplementation(() => {});
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
});

describe.each(cases)("$action", ({ action, method, endpoint, payload, base, confirmed, domain }) => {
  const listKey = domain === "disputes" ? adminDisputeKeys.list(admin.id) : payoutReportKeys.adminList(admin.id);
  const openKey = domain === "disputes" ? adminDisputeKeys.list(admin.id, { status: "open" }) : payoutReportKeys.adminList(admin.id, { status: "open" });

  it("retains the confirmed record and audit trail on failure, and displays an error", async () => {
    client.setQueryData(listKey, [confirmed]);
    const before = client.getQueryData(listKey);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    vi.spyOn(api, method).mockRejectedValueOnce(new Error("Backend unavailable"));
    const { result } = renderHook(() => useActionRunner(action), { wrapper: Wrapper });

    await act(async () => { expect(await result.current()).toBe(false); });

    expect(client.getQueryData(listKey)).toBe(before);
    expect(invalidate).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toEqual([
      expect.objectContaining({ kind: "error", message: expect.stringContaining("please try again") }),
    ]);
  });

  it("waits for the backend, replaces the complete record, and removes it from the open filter", async () => {
    client.setQueryData(listKey, [base]);
    client.setQueryData(openKey, [base]);
    const response = deferred<Dispute | PayoutReport>();
    const request = vi.spyOn(api, method).mockReturnValueOnce(response.promise);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(() => useActionRunner(action), { wrapper: Wrapper });
    let operation!: Promise<boolean>;
    act(() => { operation = result.current(); });

    expect(request).toHaveBeenCalledWith(endpoint, payload);
    expect(client.getQueryData(listKey)).toEqual([base]);
    expect(useToastStore.getState().toasts).toEqual([]);
    await act(async () => { response.resolve(confirmed); expect(await operation).toBe(true); });

    expect(client.getQueryData(listKey)).toEqual([confirmed]);
    expect(client.getQueryData(openKey)).toEqual([]);
    if (domain === "disputes") {
      expect(client.getQueryData(adminDisputeKeys.detail(admin.id, dispute.id))).toEqual(confirmed);
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "kpis"] });
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "sales"] });
    } else {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: payoutReportKeys.photographer });
    }
  });
});

it.each(["resolve dispute", "resolve report"])("cancels an older list read before applying %s, then accepts later server changes", async (action) => {
  const isDispute = action === "resolve dispute";
  const base = isDispute ? dispute : report;
  const confirmed = isDispute ? confirmedDispute : confirmedReport;
  const useRead = isDispute ? useAdminDisputes : useAdminPayoutReports;
  const oldRead = deferred<unknown>();
  vi.spyOn(api, "get").mockReturnValueOnce(oldRead.promise).mockResolvedValue({ items: [confirmed] });
  vi.spyOn(api, isDispute ? "post" : "fetch").mockResolvedValueOnce(confirmed);
  const { result } = renderHook(() => ({
    query: useRead(),
    run: useActionRunner(action),
  }), { wrapper: Wrapper });

  await waitFor(() => expect(api.get).toHaveBeenCalledTimes(1));
  await act(async () => { expect(await result.current.run()).toBe(true); });
  await waitFor(() => expect(result.current.query.data).toEqual([confirmed]));
  await act(async () => { oldRead.resolve({ items: [base] }); });
  expect(result.current.query.data).toEqual([confirmed]);

  // A later response must win; there is no persistent local override masking it.
  const later = { ...confirmed, note: "A later backend edit" };
  vi.mocked(api.get).mockResolvedValueOnce({ items: [later] });
  await act(async () => { await result.current.query.refetch(); });
  await waitFor(() => expect(result.current.query.data).toEqual([later]));
});

describe.each(["resolve dispute", "resolve report"])("stale session: %s", (action) => {
  it.each(["success", "failure", "same-user-login"])("discards a mutation after an auth transition (%s)", async (outcome) => {
    const response = deferred<Dispute | PayoutReport>();
    vi.spyOn(api, action === "resolve dispute" ? "post" : "fetch").mockReturnValueOnce(response.promise);
    const { result } = renderHook(() => useActionRunner(action), { wrapper: Wrapper });
    let operation!: Promise<boolean>;
    act(() => { operation = result.current(); });
    act(() => {
      useAuthStore.getState().logout();
      client.clear();
      if (outcome === "same-user-login") useAuthStore.getState().setUser(admin);
      else useAuthStore.getState().setUser({ ...admin, id: "admin-b" });
    });
    const nextKey = adminDisputeKeys.list(useAuthStore.getState().user!.id);
    client.setQueryData(nextKey, [{ ...dispute, id: "next-session-record" }]);
    const nextData = client.getQueryData(nextKey);
    await act(async () => {
      if (outcome === "failure") response.reject(new Error("Late failure"));
      else response.resolve(action === "resolve dispute" ? confirmedDispute : confirmedReport);
      expect(await operation).toBe(false);
    });

    expect(client.getQueryData(nextKey)).toBe(nextData);
    expect(client.getQueryCache().getAll()).toHaveLength(1);
    expect(useToastStore.getState().toasts).toEqual([]);
  });
});

it("scopes report reads to the authenticated photographer and ignores the prior account's in-flight response", async () => {
  useAuthStore.getState().setUser({ ...admin, id: "photographer-a", role: "PHOTOGRAPHER" });
  const oldRead = deferred<PaginatedResponse<PayoutReport>>();
  const newRead = deferred<PaginatedResponse<PayoutReport>>();
  vi.spyOn(api, "get").mockReturnValueOnce(oldRead.promise).mockReturnValueOnce(newRead.promise);
  const { result } = renderHook(() => usePhotographerPayoutReports(), { wrapper: Wrapper });
  await waitFor(() => expect(api.get).toHaveBeenCalledTimes(1));
  act(() => {
    useAuthStore.getState().logout();
    client.clear();
    useAuthStore.getState().setUser({ ...admin, id: "photographer-b", role: "PHOTOGRAPHER" });
  });
  await waitFor(() => expect(api.get).toHaveBeenCalledTimes(2));
  const next = { ...report, id: "report-b", photographerId: "photographer-b" };
  await act(async () => { newRead.resolve({ items: [next], total: 1, offset: 0, limit: 50 }); });
  await waitFor(() => expect(result.current.data).toEqual([next]));
  await act(async () => { oldRead.resolve({ items: [report], total: 1, offset: 0, limit: 50 }); });
  expect(result.current.data).toEqual([next]);
  expect(client.getQueryData(payoutReportKeys.photographerList("photographer-a"))).toBeUndefined();
});

it("finds a direct dispute link beyond the first page and returns null only after exhausting the list", async () => {
  vi.spyOn(api, "get")
    .mockResolvedValueOnce({ items: [{ ...dispute, id: "another-dispute" }], total: 2 })
    .mockResolvedValueOnce({ items: [dispute], total: 2 })
    .mockResolvedValueOnce({ items: [dispute], total: 1 });
  expect(await fetchAdminDispute(dispute.id)).toEqual(dispute);
  expect(api.get).toHaveBeenNthCalledWith(2, "/admin/disputes?offset=1&limit=200");
  expect(await fetchAdminDispute("missing")).toBeNull();
});

it("stops a deep-link lookup from requesting more pages after its query is cancelled", async () => {
  const response = deferred<unknown>();
  vi.spyOn(api, "get").mockReturnValueOnce(response.promise);
  const controller = new AbortController();
  const pending = fetchAdminDispute(dispute.id, controller.signal);
  controller.abort();
  response.resolve({ items: [{ ...dispute, id: "other" }], total: 2 });
  await expect(pending).rejects.toThrow();
  expect(api.get).toHaveBeenCalledTimes(1);
});
