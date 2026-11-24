"use client";

import { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminPayouts, fetchAdminPayoutsPage, fetchAdminPayoutCounts, approvePayout, holdPayout, markPayoutPaid, rejectPayout,
  markPayoutProcessing, markPayoutFailed, markPayoutReturned, bulkApprovePayouts, bulkHoldPayouts,
  type AdminPayoutListArgs, type BulkPayoutResult,
} from "@/lib/api-admin";
import type { AdminPayoutCycle, AdminPayoutStatus } from "@/lib/admin-payouts";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useAdminMutation } from "./use-admin-mutation";
import { claimAdminIntent, completeAdminIntent } from "@/lib/admin-action-intents";
import { captureAuthSession } from "@/lib/auth-session";
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
  const mutation = useAdminMutation();
    const settle = async (
      label: string,
      operation: () => Promise<AdminPayoutCycle | BulkPayoutResult>,
      onPartialFailure?: (failedIds: string[], failed: BulkPayoutResult["results"]) => void,
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
          // The server's own words for a hold refusal (PE-2.8); FT-23 (C-16): any other server refusal keeps its reason too.
          const heldRefusal = error instanceof ApiError && error.errors[0]?.code === "PAYOUT_ON_HOLD";
          const bulk = label === "Approve payouts" || label === "Hold payouts";
          useToastStore.getState().showToast({
            kind: "error", message: heldRefusal ? error.message
              : `Couldn't ${label}. ${(bulk && error instanceof Error) || error instanceof ApiError ? error.message : "Please try again."}`,
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
      await Promise.allSettled([
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
          const failedRows = result.results.filter((row) => !row.ok);
          onPartialFailure?.(failedRows.map((row) => row.id), failedRows);
          useToastStore.getState().showToast({
            kind: "error", message: `${label}: ${ok} succeeded, ${failed} failed.`,
          });
          return false;
        }
      }
      return true;
    };
    const run = async (ids: string[], action: string, label: string, operation: () => Promise<AdminPayoutCycle | BulkPayoutResult>, onPartialFailure?: (ids: string[], failed: BulkPayoutResult["results"]) => void) => {
      const result = await mutation.run({ targets: ids.map((id) => ({ kind: "payout", id })), action, execute: () => settle(label, operation, onPartialFailure) });
      return result.kind === "completed" && result.value;
    };
    const bulk = async (ids: string[], action: "approve" | "hold", reason?: string) => {
      const current = captureAuthSession();
      const actor = useAuthStore.getState().user!.id;
      const draft = { ids: [...new Set(ids)].sort(), action, reason: reason?.trim() };
      // Different batches are separate intents; persist their digest, never the reason.
      const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(draft)));
      const target = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
      if (!current()) throw new Error("Session changed.");
      const intent = await claimAdminIntent(actor, "payout.bulk", target, draft);
      if (!current()) throw new Error("Session changed.");
      try {
        const result = action === "approve" ? await bulkApprovePayouts(draft.ids, intent.key) : await bulkHoldPayouts(draft.ids, draft.reason!, intent.key);
        if (current()) completeAdminIntent(intent);
        return result;
      } catch (error) {
        throw new Error(`The batch outcome is unconfirmed. Retry the same selection, action and reason to check it.${!intent.reloadSafe ? " Keep this page open; reload recovery is unavailable." : ""}`, { cause: error });
      }
    };
    return {
      isPending: (ids: string[]) => mutation.isPending(ids.map((id) => ({ kind: "payout", id }))),
      pendingAction: (ids: string[]) => mutation.pendingAction(ids.map((id) => ({ kind: "payout", id }))),
      // FT-23 (C-15): resolves the updated payout, so the caller can tell a first approval (still pending) from the final one.
      approve: async (id: string) => {
        const done: { cycle?: AdminPayoutCycle } = {};
        const ok = await run([id], "Approving…", "approve payout", async () => (done.cycle = await approvePayout(id)));
        return ok && done.cycle ? done.cycle : false;
      },
      hold: (id: string, reason: string) => run([id], "Holding…", "hold payout", () => holdPayout(id, reason)),
      markPaid: (id: string, reference: string) => run([id], "Marking paid…", "mark payout paid", () => markPayoutPaid(id, reference)),
      reject: (id: string, reason: string) => run([id], "Rejecting…", "reject payout", () => rejectPayout(id, reason)),
      markProcessing: (id: string) => run([id], "Processing…", "mark payout processing", () => markPayoutProcessing(id)),
      markFailed: (id: string, reason: string) => run([id], "Marking failed…", "mark payout failed", () => markPayoutFailed(id, reason)),
      markReturned: (id: string, reason: string) => run([id], "Marking returned…", "mark payout returned", () => markPayoutReturned(id, reason)),
      bulkApprove: (ids: string[], onPartialFailure?: (failedIds: string[], failed: BulkPayoutResult["results"]) => void) =>
        ids.length ? run(ids, "Approving…", "Approve payouts", () => bulk(ids, "approve"), onPartialFailure) : Promise.resolve(true),
      bulkHold: (ids: string[], reason: string, onPartialFailure?: (failedIds: string[]) => void) =>
        ids.length ? run(ids, "Holding…", "Hold payouts", () => bulk(ids, "hold", reason), onPartialFailure) : Promise.resolve(true),
    };
}

// AX-30: the queue's own server lists. "queue" sits where owner() puts the user id, so
// settle()'s array patch skips these pages, while its invalidation of ["admin", "payouts"] refreshes them.
const queueKey = [...adminPayoutKeys.all, "queue"] as const;

export function useAdminPayoutsPage(status: AdminPayoutStatus[], q: string) {
  const session = usePrivateQuerySession("ADMIN");
  return useInfiniteList<AdminPayoutCycle>({
    queryKey: [...queueKey, "page", status, q, ...session.key],
    fetchPage: (offset, limit, signal) => fetchAdminPayoutsPage({ status, q, offset, limit }, signal),
    limit: 50,
    enabled: session.enabled,
    staleTime: 60_000,
  });
}

/** Per-status totals; undefined while loading or when the backend has no counts route yet. */
export function useAdminPayoutCounts(q = "") {
  const session = usePrivateQuerySession("ADMIN");
  return useQuery({
    queryKey: [...queueKey, "counts", q, ...session.key],
    queryFn: ({ signal }) => fetchAdminPayoutCounts(q, signal),
    enabled: session.enabled,
    staleTime: 60_000,
    retry: false,
  }).data;
}

// A ?row= link may name a payout on another tab or a later page: read it by id (the server q matches ids).
export function useAdminPayoutById(id: string) {
  const session = usePrivateQuerySession("ADMIN");
  return useQuery({
    queryKey: [...queueKey, "row", id, ...session.key],
    queryFn: async ({ signal }) =>
      (await fetchAdminPayoutsPage({ status: [], q: id, offset: 0, limit: 200 }, signal))
        .items.find((cycle) => cycle.id === id) ?? null,
    enabled: session.enabled && !!id,
    staleTime: 60_000,
  });
}

/** The money awaiting review, or undefined when the one 200-row read may have left cycles out. */
export function usePendingPayoutsTotal(awaiting: number | undefined): number | undefined {
  const cycles = useAdminPayouts({ status: "pending_review" }).data ?? EMPTY_PAYOUTS;
  return useMemo(() => cycles.length < 200 || cycles.length === awaiting
    ? cycles.reduce((total, cycle) => total + cycle.amount, 0) : undefined, [cycles, awaiting]);
}
