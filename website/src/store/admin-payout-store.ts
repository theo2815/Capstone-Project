import { create } from "zustand";
import {
  ADMIN_PAYOUT_SEED,
  type AdminPayoutCycle,
} from "@/lib/admin-payouts";
import { usePhotographerMessageStore } from "@/store/photographer-message-store";

// Mock-only store of admin actions on the payouts queue. NOT persisted —
// real backend ships in Phase F. Adds bulk variants on top of the
// dispute/flag pattern: bulkApprove + bulkHold emit one log entry per
// included payout id with a shared `groupId` so the activity timeline
// can collapse them visually if it wants to.
//
// Hold + bulkHold also push a "payout_held" message into the
// photographer-message-store so the affected photographer sees it in their
// inbox bell. Approve + markPaid are silent — the photographer learns from
// status flips on /dashboard/billing, not the inbox.

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
  approve: (payoutId: string) => void;
  hold: (payoutId: string, reason: string) => void;
  markPaid: (payoutId: string, reference: string) => void;
  bulkApprove: (payoutIds: string[]) => void;
  bulkHold: (payoutIds: string[], reason: string) => void;
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

export const useAdminPayoutStore = create<AdminPayoutStoreState>((set) => ({
  overrides: {},
  log: [],
  approve: (payoutId) => {
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
      log: appendLog(s.log, {
        payoutId,
        decision: "approved",
        paymentReference: null,
        reason: null,
        decidedAt: at,
      }),
    }));
  },
  hold: (payoutId, reason) => {
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
      log: appendLog(s.log, {
        payoutId,
        decision: "held",
        paymentReference: null,
        reason,
        decidedAt: at,
      }),
    }));
    const cycle = ADMIN_PAYOUT_SEED.find((p) => p.id === payoutId);
    if (cycle) {
      usePhotographerMessageStore.getState().addMessage({
        photographerId: cycle.photographerId,
        kind: "payout_held",
        title: "Payout held — review pending",
        body: reason,
        payoutCycleId: payoutId,
      });
    }
  },
  markPaid: (payoutId, reference) => {
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
      log: appendLog(s.log, {
        payoutId,
        decision: "paid",
        paymentReference: reference,
        reason: null,
        decidedAt: at,
      }),
    }));
  },
  bulkApprove: (payoutIds) => {
    if (payoutIds.length === 0) return;
    const at = nowIso();
    const grp = groupId();
    set((s) => {
      const overrides = { ...s.overrides };
      const entries: PayoutLogEntry[] = [];
      for (const id of payoutIds) {
        overrides[id] = {
          ...overrides[id],
          status: "approved",
          reviewedAt: at,
          holdReason: null,
        };
        entries.push({
          payoutId: id,
          decision: "bulk_approved",
          paymentReference: null,
          reason: null,
          decidedAt: at,
          groupId: grp,
        });
      }
      return { overrides, log: appendLog(s.log, entries) };
    });
  },
  bulkHold: (payoutIds, reason) => {
    if (payoutIds.length === 0) return;
    const at = nowIso();
    const grp = groupId();
    set((s) => {
      const overrides = { ...s.overrides };
      const entries: PayoutLogEntry[] = [];
      for (const id of payoutIds) {
        overrides[id] = {
          ...overrides[id],
          status: "held",
          reviewedAt: at,
          holdReason: reason,
        };
        entries.push({
          payoutId: id,
          decision: "bulk_held",
          paymentReference: null,
          reason,
          decidedAt: at,
          groupId: grp,
        });
      }
      return { overrides, log: appendLog(s.log, entries) };
    });
    const inbox = usePhotographerMessageStore.getState();
    for (const id of payoutIds) {
      const cycle = ADMIN_PAYOUT_SEED.find((p) => p.id === id);
      if (!cycle) continue;
      inbox.addMessage({
        photographerId: cycle.photographerId,
        kind: "payout_held",
        title: "Payout held — review pending",
        body: reason,
        payoutCycleId: id,
      });
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
