import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, dispute, payoutCycle, report } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { useFlagActions } from "./use-admin-data";
import { useDisputeActions } from "./use-admin-disputes";
import { usePayoutActions } from "./use-admin-payouts";
import { usePayoutReportActions } from "./use-payout-reports";

let client: QueryClient;
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

function useFamily(family: string) {
  const flags = useFlagActions();
  const disputes = useDisputeActions();
  const payouts = usePayoutActions();
  const reports = usePayoutReportActions();
  if (family === "flag") return { first: () => flags.hide("flag-1", "Reason"), other: () => flags.dismiss("flag-1") };
  if (family === "dispute") return { first: () => disputes.deny(dispute.id, "Reason"), other: () => disputes.escalate(dispute.id, "Reason") };
  if (family === "payout") return { first: () => payouts.approve(payoutCycle.id), other: () => payouts.hold(payoutCycle.id, "Reason") };
  return { first: () => reports.acknowledge(report.id, "Reply"), other: () => reports.resolve(report.id, "Note") };
}

it.each(["flag", "dispute", "payout", "report"])("%s excludes same-turn conflicting commands across views and releases after rejection", async (family) => {
  let reject!: (error: Error) => void;
  const request = new Promise((_, no) => { reject = no; });
  const transport = vi.spyOn(api, family === "report" ? "fetch" : "post").mockReturnValue(request);
  const a = renderHook(() => useFamily(family), { wrapper });
  const b = renderHook(() => useFamily(family), { wrapper });
  let first!: Promise<unknown>, blocked!: Promise<unknown>;
  act(() => { first = a.result.current.first().catch(() => false); blocked = b.result.current.other(); });
  await waitFor(() => expect(transport).toHaveBeenCalledTimes(1));
  await expect(blocked).resolves.toBe(false);
  await act(async () => { reject(new Error("Unavailable")); await first; });
  transport.mockResolvedValue(family === "dispute" ? dispute : family === "payout" ? payoutCycle : report);
  await act(async () => { expect(await b.result.current.other()).toBe(true); });
  expect(transport).toHaveBeenCalledTimes(2);
});

it("bulk payout reserves every row before hashing and reuses the original key after a lost response", async () => {
  const post = vi.spyOn(api, "post").mockRejectedValue(new Error("Response lost"));
  const a = renderHook(usePayoutActions, { wrapper });
  const b = renderHook(usePayoutActions, { wrapper });
  let bulk!: Promise<boolean>, row!: Promise<boolean>;
  act(() => { bulk = a.result.current.bulkApprove(["b", "a"]); row = b.result.current.hold("a", "Review"); });
  await expect(row).resolves.toBe(false);
  await expect(bulk).resolves.toBe(false);
  expect(post).toHaveBeenCalledTimes(1);
  const original = post.mock.calls[0]![2];
  a.unmount();
  await act(async () => { await b.result.current.bulkApprove(["a", "b"]); });
  expect(post.mock.calls[1]![2]).toEqual(original);
  expect(post.mock.calls[1]![1]).toEqual({ ids: ["a", "b"], action: "approve" });
  post.mockResolvedValue({ groupId: "batch", results: [{ id: "a", ok: true }, { id: "b", ok: false }] });
  const failed = vi.fn();
  await act(async () => { await b.result.current.bulkApprove(["b", "a"], failed); });
  expect(failed).toHaveBeenCalledWith(["b"]);
  await act(async () => { await b.result.current.bulkApprove(["b"]); });
  expect(post.mock.calls[3]![2]).not.toEqual(original);
});

it.each(["approve", "hold"])("bulk %s preserves its key for reordered identical retries", async (action) => {
  const post = vi.spyOn(api, "post").mockRejectedValue(new Error("Response lost"));
  const { result } = renderHook(usePayoutActions, { wrapper });
  const send = (ids: string[]) => action === "approve" ? result.current.bulkApprove(ids) : result.current.bulkHold(ids, " Review ");
  await act(async () => { await send(["b", "a"]); await send(["a", "b"]); });
  expect(post.mock.calls[1]![2]).toEqual(post.mock.calls[0]![2]);
});

it("explains how to retry an uncertain payout batch", async () => {
  vi.spyOn(api, "post").mockRejectedValue(new Error("Response lost"));
  const { result } = renderHook(usePayoutActions, { wrapper });
  await act(async () => { await result.current.bulkHold(["a"], "Review"); });
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("same selection, action and reason");
});

it.each(["flag", "dispute", "payout", "report"])("%s does not turn an accepted command into a retryable failure when refresh fails", async (family) => {
  vi.spyOn(api, family === "report" ? "fetch" : "post").mockResolvedValue(family === "dispute" ? dispute : family === "payout" ? payoutCycle : report);
  vi.spyOn(client, "invalidateQueries").mockRejectedValue(new Error("Refresh unavailable"));
  const { result } = renderHook(() => useFamily(family), { wrapper });
  await act(async () => { await expect(result.current.first()).resolves.toBe(true); });
  expect(useToastStore.getState().toasts.filter((toast) => toast.kind === "error")).toHaveLength(0);
});
