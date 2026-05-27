import { create } from "zustand";
import {
  ADMIN_PAYOUT_SEED,
  type AdminPayoutCycle,
} from "@/lib/admin-payouts";
import {
  approvePayout as apiApprovePayout,
  holdPayout as apiHoldPayout,
  markPayoutPaid as apiMarkPayoutPaid,
  bulkApprovePayouts as apiBulkApprovePayouts,
  bulkHoldPayouts as apiBulkHoldPayouts,
} from "@/lib/api-admin";
import { useToastStore } from "@/store/toast-store";

// Mirror of admin-user-store: each mutation awaits the BE, appends to the
// decision log only on success, and rolls back the optimistic override on
// failure with a toast. Photographer notifications fire BE-side
// (AdminPayoutService → AdminDecisionLogService.pushMessage in the same
// TX; kinds payout_held / payout_approved / payout_paid). Photographer
// reads them via GET /me/photographer/messages.
//
// Bulk actions parse BulkPayoutResult.results so per-id BE failures
// (200 OK + ok:false) roll back individually — full rejection rolls back
// the whole batch.

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
  | "held"
  | "paid"
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

function nowIso() {
  return new Date().toISOString();
}

function groupId() {
  return `grp-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 6)}`;
}

// Restore a single override slot to its prior value (or delete the slot
// when there was no prior entry). Used by the rollback paths below.
function rollbackOverride(
  overrides: Record<string, Partial<AdminPayoutCycle>>,
  payoutId: string,
  prior: Partial<AdminPayoutCycle> | undefined,
): Record<string, Partial<AdminPayoutCycle>> {
  const next = { ...overrides };
  if (prior === undefined) delete next[payoutId];
  else next[payoutId] = prior;
  return next;
}

export const useAdminPayoutStore = create<AdminPayoutStoreState>((set, get) => ({
  overrides: {},
  log: [],
  approve: async (payoutId) => {
    const prior = get().overrides[payoutId];
    const at = nowIso();
    set((s) => ({
      overrides: {
        ...s.overrides,
        [payoutId]: {
          ...s.overrides[payoutId],
          status: "approved",
          reviewedAt: at,
          holdReason: null,
        },
      },
    }));
    try {
      await apiApprovePayout(payoutId);
      set((s) => ({
        log: appendLog(s.log, {
          payoutId,
          decision: "approved",
          paymentReference: null,
          reason: null,
          decidedAt: at,
        }),
      }));
      return true;
    } catch (err) {
      set((s) => ({ overrides: rollbackOverride(s.overrides, payoutId, prior) }));
      toastBackendFailure("approve payout", err);
      return false;
    }
  },
  hold: async (payoutId, reason) => {
    const prior = get().overrides[payoutId];
    const at = nowIso();
    set((s) => ({
      overrides: {
        ...s.overrides,
        [payoutId]: {
          ...s.overrides[payoutId],
          status: "held",
          reviewedAt: at,
          holdReason: reason,
        },
      },
    }));
    try {
      await apiHoldPayout(payoutId, reason);
      set((s) => ({
        log: appendLog(s.log, {
          payoutId,
          decision: "held",
          paymentReference: null,
          reason,
          decidedAt: at,
        }),
      }));
      return true;
    } catch (err) {
      set((s) => ({ overrides: rollbackOverride(s.overrides, payoutId, prior) }));
      toastBackendFailure("hold payout", err);
      return false;
    }
  },
  markPaid: async (payoutId, reference) => {
    const prior = get().overrides[payoutId];
    const at = nowIso();
    set((s) => ({
      overrides: {
        ...s.overrides,
        [payoutId]: {
          ...s.overrides[payoutId],
          status: "paid",
          paidAt: at,
          paymentReference: reference,
        },
      },
    }));
    try {
      await apiMarkPayoutPaid(payoutId, reference);
      set((s) => ({
        log: appendLog(s.log, {
          payoutId,
          decision: "paid",
          paymentReference: reference,
          reason: null,
          decidedAt: at,
        }),
      }));
      return true;
    } catch (err) {
      set((s) => ({ overrides: rollbackOverride(s.overrides, payoutId, prior) }));
      toastBackendFailure("mark payout paid", err);
      return false;
    }
  },
  bulkApprove: async (payoutIds) => {
    if (payoutIds.length === 0) return true;
    const priors = new Map<string, Partial<AdminPayoutCycle> | undefined>();
    for (const id of payoutIds) priors.set(id, get().overrides[id]);
    const at = nowIso();
    const grp = groupId();
    set((s) => {
      const overrides = { ...s.overrides };
      for (const id of payoutIds) {
        overrides[id] = {
          ...overrides[id],
          status: "approved",
          reviewedAt: at,
          holdReason: null,
        };
      }
      return { overrides };
    });
    try {
      const result = await apiBulkApprovePayouts(payoutIds);
      const okIds = result.results.filter((r) => r.ok).map((r) => r.id);
      const failedIds = result.results.filter((r) => !r.ok).map((r) => r.id);
      const logEntries: PayoutLogEntry[] = okIds.map((id) => ({
        payoutId: id,
        decision: "bulk_approved",
        paymentReference: null,
        reason: null,
        decidedAt: at,
        groupId: result.groupId ?? grp,
      }));
      set((s) => {
        let overrides = s.overrides;
        for (const id of failedIds) {
          overrides = rollbackOverride(overrides, id, priors.get(id));
        }
        return {
          overrides,
          log: logEntries.length > 0 ? appendLog(s.log, logEntries) : s.log,
        };
      });
      if (failedIds.length === 0) return true;
      if (okIds.length === 0) {
        toastBackendFailure("approve payouts", new Error("All ids failed"));
      } else {
        toastBulkPartial("Approve payouts", okIds.length, failedIds.length);
      }
      return false;
    } catch (err) {
      set((s) => {
        let overrides = s.overrides;
        for (const id of payoutIds) {
          overrides = rollbackOverride(overrides, id, priors.get(id));
        }
        return { overrides };
      });
      toastBackendFailure("approve payouts", err);
      return false;
    }
  },
  bulkHold: async (payoutIds, reason) => {
    if (payoutIds.length === 0) return true;
    const priors = new Map<string, Partial<AdminPayoutCycle> | undefined>();
    for (const id of payoutIds) priors.set(id, get().overrides[id]);
    const at = nowIso();
    const grp = groupId();
    set((s) => {
      const overrides = { ...s.overrides };
      for (const id of payoutIds) {
        overrides[id] = {
          ...overrides[id],
          status: "held",
          reviewedAt: at,
          holdReason: reason,
        };
      }
      return { overrides };
    });
    try {
      const result = await apiBulkHoldPayouts(payoutIds, reason);
      const okIds = result.results.filter((r) => r.ok).map((r) => r.id);
      const failedIds = result.results.filter((r) => !r.ok).map((r) => r.id);
      const logEntries: PayoutLogEntry[] = okIds.map((id) => ({
        payoutId: id,
        decision: "bulk_held",
        paymentReference: null,
        reason,
        decidedAt: at,
        groupId: result.groupId ?? grp,
      }));
      set((s) => {
        let overrides = s.overrides;
        for (const id of failedIds) {
          overrides = rollbackOverride(overrides, id, priors.get(id));
        }
        return {
          overrides,
          log: logEntries.length > 0 ? appendLog(s.log, logEntries) : s.log,
        };
      });
      if (failedIds.length === 0) return true;
      if (okIds.length === 0) {
        toastBackendFailure("hold payouts", new Error("All ids failed"));
      } else {
        toastBulkPartial("Hold payouts", okIds.length, failedIds.length);
      }
      return false;
    } catch (err) {
      set((s) => {
        let overrides = s.overrides;
        for (const id of payoutIds) {
          overrides = rollbackOverride(overrides, id, priors.get(id));
        }
        return { overrides };
      });
      toastBackendFailure("hold payouts", err);
      return false;
    }
  },
  clear: () => set({ overrides: {}, log: [] }),
}));

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
 * Merges live server data with local overrides for instant feedback after
 * admin actions. Use this in components that call `useAdminPayouts()` directly.
 */
export function mergePayoutsWithOverrides(
  serverData: ReadonlyArray<AdminPayoutCycle>,
  overrides: Record<string, Partial<AdminPayoutCycle>>,
): AdminPayoutCycle[] {
  return serverData.map((p) => mergePayout(p, overrides[p.id]));
}
