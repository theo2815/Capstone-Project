"use client";

import { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminPayouts, approvePayout, holdPayout, markPayoutPaid, rejectPayout,
  markPayoutProcessing, markPayoutFailed, markPayoutReturned, bulkApprovePayouts, bulkHoldPayouts,
  type AdminPayoutListArgs, type BulkPayoutResult,
} from "@/lib/api-admin";
import type { AdminPayoutCycle } from "@/lib/admin-payouts";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";

export const EMPTY_PAYOUTS: AdminPayoutCycle[] = [];
export const adminPayoutKeys = {
  // Keep cycle rows separate from ["admin", "payouts", "reports"].
  all: ["admin", "payouts", "cycles"] as const,
  owner: (userId: string | null) => [...adminPayoutKeys.all, userId] as const,
  list: (userId: string | null, args: AdminPayoutListArgs = {}) =>
    [...adminPayoutKeys.owner(userId), args] as const,
};

export function useAdminPayouts(args: AdminPayoutListArgs = {}) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminPayoutKeys.list(user?.id ?? null, args),
    queryFn: () => fetchAdminPayouts(args),
    enabled: user?.role === "ADMIN",
    staleTime: 60_000,
  });
}

export function usePayoutActions() {
  const qc = useQueryClient();
  return useMemo(() => {
    const settle = async (
      label: string,
      operation: () => Promise<AdminPayoutCycle | BulkPayoutResult>,
      onPartialFailure?: (failedIds: string[]) => void,
    ): Promise<boolean> => {
      const session = useAuthStore.getState();
      if (!session.isAuthenticated || session.user?.role !== "ADMIN") return false;
      const userId = session.user.id;
      const isCurrent = () => {
        const current = useAuthStore.getState();
        return current.isAuthenticated && current.user?.role === "ADMIN" &&
          current.user.id === userId && current.sessionVersion === session.sessionVersion;
      };
      let result: AdminPayoutCycle | BulkPayoutResult;
      try {
        // The API client owns MFA/step-up and retries. This promise stays pending
        // through that exchange; the caller must keep its controls disabled.
        result = await operation();
      } catch (error) {
        if (isCurrent()) {
          console.error(`[admin/payouts] ${label} failed`, error);
          useToastStore.getState().showToast({
            kind: "error", message: `Couldn't ${label} — please try again.`,
          });
        }
        return false;
      }
      if (!isCurrent()) return false;
      await qc.cancelQueries({ queryKey: adminPayoutKeys.owner(userId) });
      if (!isCurrent()) return false;
      if (!("results" in result)) {
        const cycle = result;
        for (const [key] of qc.getQueriesData<AdminPayoutCycle[]>({ queryKey: adminPayoutKeys.owner(userId) })) {
          const args = key.at(-1) as AdminPayoutListArgs;
          qc.setQueryData<AdminPayoutCycle[]>(key, (rows) => rows?.flatMap((row) => {
            if (row.id !== cycle.id) return [row];
            return args.status && args.status !== cycle.status ? [] : [cycle];
          }));
        }
      }
      // Bulk results contain only per-ID outcomes, not records. Refetch even on
      // partial success; never infer approval status (the first may stay pending).
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["admin", "payouts"] }),
        qc.invalidateQueries({ queryKey: ["admin", "kpis"] }),
        qc.invalidateQueries({ queryKey: ["admin", "sales"] }),
      ]);
      if (!isCurrent()) return false;
      if ("results" in result) {
        const ok = result.results.filter((row) => row.ok).length;
        const failed = result.results.length - ok;
        if (failed) {
          // Selection is UI state. Let the caller retain only retryable IDs,
          // including first-approval successes that still look pending.
          onPartialFailure?.(result.results.filter((row) => !row.ok).map((row) => row.id));
          useToastStore.getState().showToast({
            kind: "error", message: `${label}: ${ok} succeeded, ${failed} failed.`,
          });
          return false;
        }
      }
      return true;
    };
    return {
      approve: (id: string) => settle("approve payout", () => approvePayout(id)),
      hold: (id: string, reason: string) => settle("hold payout", () => holdPayout(id, reason)),
      markPaid: (id: string, reference: string) => settle("mark payout paid", () => markPayoutPaid(id, reference)),
      reject: (id: string, reason: string) => settle("reject payout", () => rejectPayout(id, reason)),
      markProcessing: (id: string) => settle("mark payout processing", () => markPayoutProcessing(id)),
      markFailed: (id: string, reason: string) => settle("mark payout failed", () => markPayoutFailed(id, reason)),
      markReturned: (id: string, reason: string) => settle("mark payout returned", () => markPayoutReturned(id, reason)),
      bulkApprove: (ids: string[], onPartialFailure?: (failedIds: string[]) => void) =>
        ids.length ? settle("Approve payouts", () => bulkApprovePayouts(ids), onPartialFailure) : Promise.resolve(true),
      bulkHold: (ids: string[], reason: string, onPartialFailure?: (failedIds: string[]) => void) =>
        ids.length ? settle("Hold payouts", () => bulkHoldPayouts(ids, reason), onPartialFailure) : Promise.resolve(true),
    };
  }, [qc]);
}

export function usePendingPayoutsCount(): number {
  const cycles = useAdminPayouts().data ?? EMPTY_PAYOUTS;
  return useMemo(() => cycles.filter((cycle) => cycle.status === "pending_review").length, [cycles]);
}

export function usePendingPayoutsTotal(): number {
  const cycles = useAdminPayouts().data ?? EMPTY_PAYOUTS;
  return useMemo(() => cycles.filter((cycle) => cycle.status === "pending_review")
    .reduce((total, cycle) => total + cycle.amount, 0), [cycles]);
}
