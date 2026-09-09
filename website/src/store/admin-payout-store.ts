import { create } from "zustand";
import {
  ADMIN_PAYOUT_SEED,
  type AdminPayoutCycle,
} from "@/lib/admin-payouts";
import {
  approvePayout as apiApprovePayout,
  holdPayout as apiHoldPayout,
  markPayoutPaid as apiMarkPayoutPaid,
  rejectPayout as apiRejectPayout,
  markPayoutProcessing as apiMarkPayoutProcessing,
  markPayoutFailed as apiMarkPayoutFailed,
  markPayoutReturned as apiMarkPayoutReturned,
  bulkApprovePayouts as apiBulkApprovePayouts,
  bulkHoldPayouts as apiBulkHoldPayouts,
} from "@/lib/api-admin";
import { useToastStore } from "@/store/toast-store";

// Financial state comes from the server. usePayoutActions refreshes the queue
// and KPIs after decisions, including a first approval that stays pending.

function toastBackendFailure(label: string, err: unknown): void {
  console.error(`[admin/payouts] ${label} failed`, err);
  useToastStore.getState().showToast({
    kind: "error",
    message: `Couldn't ${label} — please try again.`,
  });
}

function toastBulkPartial(label: string, ok: number, failed: number): void {
  useToastStore.getState().showToast({
    kind: "error",
    message: `${label}: ${ok} succeeded, ${failed} failed.`,
  });
}

export type PayoutDecision =
  | "approved"
  | "approval_recorded"
  | "held"
  | "paid"
  | "rejected"
  | "processing"
  | "failed"
  | "returned"
  | "bulk_approved"
  | "bulk_held";

export interface PayoutLogEntry {
  payoutId: string;
  decision: PayoutDecision;
  paymentReference: string | null;
  reason: string | null;
  decidedAt: string;
  groupId?: string;
}

interface AdminPayoutStoreState {
  overrides: Record<string, Partial<AdminPayoutCycle>>;
  log: PayoutLogEntry[];
  approve: (payoutId: string) => Promise<boolean>;
  hold: (payoutId: string, reason: string) => Promise<boolean>;
  markPaid: (payoutId: string, reference: string) => Promise<boolean>;
  reject: (payoutId: string, reason: string) => Promise<boolean>;
  markProcessing: (payoutId: string) => Promise<boolean>;
  markFailed: (payoutId: string, reason: string) => Promise<boolean>;
  markReturned: (payoutId: string, reason: string) => Promise<boolean>;
  bulkApprove: (payoutIds: string[]) => Promise<boolean>;
  bulkHold: (payoutIds: string[], reason: string) => Promise<boolean>;
  clear: () => void;
}

function appendLog(
  prev: PayoutLogEntry[],
  entries: PayoutLogEntry[] | PayoutLogEntry,
): PayoutLogEntry[] {
  const arr = Array.isArray(entries) ? entries : [entries];
  return [...arr, ...prev].slice(0, 50);
}

export const useAdminPayoutStore = create<AdminPayoutStoreState>((set) => {
  async function decide(
    payoutId: string,
    decision: PayoutDecision,
    label: string,
    operation: Promise<AdminPayoutCycle>,
    reason: string | null = null,
    paymentReference: string | null = null,
  ): Promise<boolean> {
    try {
      const cycle = await operation;
      set((s) => ({
        log: appendLog(s.log, {
          payoutId,
          decision: decision === "approved" && cycle.status !== "approved" ? "approval_recorded" : decision,
          reason,
          paymentReference,
          decidedAt: new Date().toISOString(),
        }),
      }));
      return true;
    } catch (err) {
      toastBackendFailure(label, err);
      return false;
    }
  }

  async function bulkDecide(
    decision: "bulk_approved" | "bulk_held",
    label: string,
    operation: ReturnType<typeof apiBulkApprovePayouts>,
    reason: string | null = null,
  ): Promise<boolean> {
    try {
      const result = await operation;
      const ok = result.results.filter((r) => r.ok);
      const failed = result.results.length - ok.length;
      set((s) => ({
        log: appendLog(s.log, ok.map(({ id }) => ({
          payoutId: id, decision, reason, paymentReference: null,
          decidedAt: new Date().toISOString(), groupId: result.groupId,
        }))),
      }));
      if (failed) toastBulkPartial(label, ok.length, failed);
      return failed === 0;
    } catch (err) {
      toastBackendFailure(label, err);
      return false;
    }
  }

  return {
    // Kept for existing read consumers; money/status overrides are never created.
    overrides: {},
    log: [],
    approve: (id) => decide(id, "approved", "approve payout", apiApprovePayout(id)),
    hold: (id, reason) => decide(id, "held", "hold payout", apiHoldPayout(id, reason), reason),
    markPaid: (id, reference) => decide(id, "paid", "mark payout paid", apiMarkPayoutPaid(id, reference), null, reference),
    reject: (id, reason) => decide(id, "rejected", "reject payout", apiRejectPayout(id, reason), reason),
    markProcessing: (id) => decide(id, "processing", "mark payout processing", apiMarkPayoutProcessing(id)),
    markFailed: (id, reason) => decide(id, "failed", "mark payout failed", apiMarkPayoutFailed(id, reason), reason),
    markReturned: (id, reason) => decide(id, "returned", "mark payout returned", apiMarkPayoutReturned(id, reason), reason),
    bulkApprove: (ids) => ids.length ? bulkDecide("bulk_approved", "Approve payouts", apiBulkApprovePayouts(ids)) : Promise.resolve(true),
    bulkHold: (ids, reason) => ids.length ? bulkDecide("bulk_held", "Hold payouts", apiBulkHoldPayouts(ids, reason), reason) : Promise.resolve(true),
    clear: () => set({ overrides: {}, log: [] }),
  };
});

export function mergePayout(
  base: AdminPayoutCycle,
  patch: Partial<AdminPayoutCycle> | undefined,
): AdminPayoutCycle {
  return patch ? { ...base, ...patch } : base;
}

export function getEffectivePayouts(
  overrides: Record<string, Partial<AdminPayoutCycle>>,
): AdminPayoutCycle[] {
  return ADMIN_PAYOUT_SEED.map((p) => mergePayout(p, overrides[p.id]));
}

/**
 * Legacy consumers still pass overrides; live payout decisions never create them.
 */
export function mergePayoutsWithOverrides(
  serverData: ReadonlyArray<AdminPayoutCycle>,
  overrides: Record<string, Partial<AdminPayoutCycle>>,
): AdminPayoutCycle[] {
  return serverData.map((p) => mergePayout(p, overrides[p.id]));
}
