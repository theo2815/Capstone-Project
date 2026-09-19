import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { AdminPayoutCycle } from "@/lib/admin-payouts";
import { admin, payoutCycle, report } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { payoutReportKeys } from "./use-payout-reports";
import { adminPayoutKeys, useAdminPayouts, usePayoutActions } from "./use-admin-payouts";

let client: QueryClient;
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const key = adminPayoutKeys.list(admin.id);
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(key, [payoutCycle]);
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => {
  cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks();
});

const actions = [
  ["approve", "approved"], ["hold", "held"], ["markPaid", "paid"],
  ["reject", "rejected"], ["markProcessing", "processing"],
  ["markFailed", "failed"], ["markReturned", "returned"],
] as const;

it.each([...actions.map(([name]) => name), "bulkApprove", "bulkHold"] as const)(
  "%s leaves canonical rows untouched while pending and on failure", async (action) => {
    const request = deferred<AdminPayoutCycle>();
    const post = vi.spyOn(api, "post").mockReturnValueOnce(request.promise);
    const { result } = renderHook(usePayoutActions, { wrapper });
    let pending!: Promise<boolean>;
    act(() => {
      pending = action === "bulkApprove" ? result.current.bulkApprove([payoutCycle.id])
        : action === "bulkHold" ? result.current.bulkHold([payoutCycle.id], "Review needed")
          : result.current[action](payoutCycle.id, "Review needed");
    });
    expect(client.getQueryData(key)).toEqual([payoutCycle]);
    expect(useToastStore.getState().toasts).toHaveLength(0);
    request.reject(new Error("Unavailable"));
    await expect(pending).resolves.toBe(false);
    expect(post).toHaveBeenCalledTimes(1);
    expect(client.getQueryData(key)).toEqual([payoutCycle]);
    expect(useToastStore.getState().toasts.at(-1)).toMatchObject({ kind: "error" });
  },
);

it.each(actions)("%s uses the full backend record and reconciles filtered caches", async (action, status) => {
  const confirmed = { ...payoutCycle, status, reviewedAt: "2026-09-10T05:00:00Z", approvedBy: "server-admin", paymentReference: "SERVER-REFERENCE" };
  const filtered = adminPayoutKeys.list(admin.id, { status: "pending_review" });
  client.setQueryData(filtered, [payoutCycle]);
  const reportKey = payoutReportKeys.adminList(admin.id);
  client.setQueryData(reportKey, [report]);
  const otherOwner = adminPayoutKeys.list("admin-b");
  client.setQueryData(otherOwner, [payoutCycle]);
  vi.spyOn(api, "post").mockResolvedValue(confirmed);
  const { result } = renderHook(usePayoutActions, { wrapper });
  await expect(result.current[action](payoutCycle.id, "User input")).resolves.toBe(true);
  expect(client.getQueryData(key)).toEqual([confirmed]);
  expect(client.getQueryData(filtered)).toEqual([]);
  expect(client.getQueryData(reportKey)).toEqual([report]);
  expect(client.getQueryData(otherOwner)).toEqual([payoutCycle]);
  expect(client.getQueryState(reportKey)?.isInvalidated).toBe(true);
});

it.each(["bulkApprove", "bulkHold"] as const)("%s refetches partial results without fabricating cycle states", async (action) => {
  const second = { ...payoutCycle, id: "cycle-2" };
  client.setQueryData(key, [payoutCycle, second]);
  const refreshed = deferred<{ items: AdminPayoutCycle[] }>();
  vi.spyOn(api, "get").mockReturnValue(refreshed.promise);
  const post = vi.spyOn(api, "post").mockResolvedValue({ groupId: "batch", results: [
    { id: payoutCycle.id, ok: true }, { id: second.id, ok: false, error: "Conflict" },
  ] });
  const { result } = renderHook(() => ({ query: useAdminPayouts(), actions: usePayoutActions() }), { wrapper });
  let pending!: Promise<boolean>;
  const onPartialFailure = vi.fn();
  act(() => { pending = action === "bulkApprove"
    ? result.current.actions.bulkApprove([payoutCycle.id, second.id], onPartialFailure)
    : result.current.actions.bulkHold([payoutCycle.id, second.id], "Needs review", onPartialFailure); });
  await waitFor(() => expect(api.get).toHaveBeenCalled());
  // A bulk response supplies no DTO and may represent only the first approval.
  expect(client.getQueryData(key)).toEqual([payoutCycle, second]);
  const confirmed = { ...payoutCycle, approvedBy: admin.id, status: action === "bulkHold" ? "held" as const : "pending_review" as const };
  await act(async () => { refreshed.resolve({ items: [confirmed, second] }); expect(await pending).toBe(false); });
  await waitFor(() => expect(result.current.query.data).toEqual([confirmed, second]));
  expect(useToastStore.getState().toasts.at(-1)?.message).toContain("1 succeeded, 1 failed");
  expect(onPartialFailure).toHaveBeenCalledWith([second.id]);
  expect(post).toHaveBeenCalledWith("/admin/payouts/bulk", expect.objectContaining({ ids: [payoutCycle.id, second.id] }),
    { headers: { "Idempotency-Key": expect.any(String) } });
});

it("cancels an older read and lets a subsequent provider refetch replace the confirmed record", async () => {
  const older = deferred<AdminPayoutCycle[]>();
  const read = client.fetchQuery({ queryKey: key, queryFn: () => older.promise }).catch(() => {});
  const approved = { ...payoutCycle, status: "approved" as const, approvedBy: admin.id };
  vi.spyOn(api, "post").mockResolvedValue(approved);
  const { result } = renderHook(usePayoutActions, { wrapper });
  await expect(result.current.approve(payoutCycle.id)).resolves.toBe(true);
  older.resolve([payoutCycle]); await read;
  expect(client.getQueryData(key)).toEqual([approved]);
  const paid = { ...approved, status: "paid" as const, paymentReference: "PROVIDER-PAID" };
  vi.spyOn(api, "get").mockResolvedValue({ items: [paid] });
  const query = renderHook(useAdminPayouts, { wrapper });
  await waitFor(() => expect(query.result.current.data).toEqual([paid]));
});

it.each(["logout", "account-change", "same-user-relogin"])("discards late single and bulk outcomes after %s", async (transition) => {
  const request = deferred<AdminPayoutCycle>();
  const bulk = deferred<unknown>();
  vi.spyOn(api, "post").mockReturnValueOnce(request.promise).mockReturnValueOnce(bulk.promise);
  const { result } = renderHook(usePayoutActions, { wrapper });
  const pending = result.current.approve(payoutCycle.id);
  const onPartialFailure = vi.fn();
  const pendingBulk = result.current.bulkApprove([payoutCycle.id], onPartialFailure);
  useAuthStore.getState().logout(); client.clear();
  if (transition !== "logout") useAuthStore.getState().setUser({ ...admin, id: transition === "account-change" ? "admin-b" : admin.id });
  const newKey = adminPayoutKeys.list(useAuthStore.getState().user?.id ?? null);
  client.setQueryData(newKey, []);
  request.resolve({ ...payoutCycle, status: "approved" });
  bulk.resolve({ groupId: "old-batch", results: [{ id: payoutCycle.id, ok: false }] });
  await expect(pending).resolves.toBe(false);
  await expect(pendingBulk).resolves.toBe(false);
  expect(onPartialFailure).not.toHaveBeenCalled();
  expect(client.getQueryData(newKey)).toEqual([]);
  expect(client.getQueryState(newKey)?.isInvalidated).toBe(false);
  if (transition !== "same-user-relogin") expect(client.getQueryData(key)).toBeUndefined();
  expect(useToastStore.getState().toasts).toHaveLength(0);
});

it("suppresses a late failure toast after logout", async () => {
  const request = deferred<AdminPayoutCycle>();
  vi.spyOn(api, "post").mockReturnValue(request.promise);
  const { result } = renderHook(usePayoutActions, { wrapper });
  const pending = result.current.markPaid(payoutCycle.id, "REF-123");
  useAuthStore.getState().logout(); client.clear();
  request.reject(new Error("Old session failure"));
  await expect(pending).resolves.toBe(false);
  expect(useToastStore.getState().toasts).toHaveLength(0);
});

it("does not read or mutate payout cycles without an admin session", async () => {
  useAuthStore.getState().logout(); client.clear();
  const get = vi.spyOn(api, "get"); const post = vi.spyOn(api, "post");
  const { result } = renderHook(() => ({ query: useAdminPayouts(), actions: usePayoutActions() }), { wrapper });
  expect(result.current.query.fetchStatus).toBe("idle");
  await expect(result.current.actions.approve(payoutCycle.id)).resolves.toBe(false);
  await expect(result.current.actions.bulkHold([payoutCycle.id], "Note")).resolves.toBe(false);
  expect(get).not.toHaveBeenCalled(); expect(post).not.toHaveBeenCalled();
});
