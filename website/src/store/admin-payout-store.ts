import { create } from "zustand";
import {
  ADMIN_PAYOUT_SEED,
  type AdminPayoutCycle,
} from "@/lib/admin-payouts";
import { usePhotographerMessageStore } from "@/store/photographer-message-store";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  approvePayout as apiApprovePayout,
  holdPayout as apiHoldPayout,
  markPayoutPaid as apiMarkPayoutPaid,
  bulkApprovePayouts as apiBulkApprovePayouts,
  bulkHoldPayouts as apiBulkHoldPayouts,
} from "@/lib/api-admin";

// Mock-mode store of admin actions on the payouts queue. Phase G wiring
// fires the matching `api-admin` call in the background after the local
// override is applied — Q-A1 + Q-A2 + Q-A4 RESOLVED 2026-05-09 means cycle
// IDs follow `PAY-{YYYY}W{NN}-{HANDLE}` format, photographer-side
// `payout_held` inbox row is written by the backend in the same TX, and
// bulk operations carry a server-assigned `group_id` for KPI collapse.
//
// The local groupId helper below stays as the mock-mode fallback; in live
// mode the FE still uses it for visual collapse until the backend response
// supplies a canonical UUID.
//
// Hold + bulkHold also push a "payout_held" message into the
// photographer-message-store so the affected photographer sees it in their
// inbox bell. Approve + markPaid are silent — the photographer learns from
// status flips on /dashboard/billing, not the inbox.

function fireBackendPayoutAction(label: string, p: Promise<unknown>): void {
  if (!BACKEND_LIVE) return;
  void p.catch((err) => {
    console.error(`[admin/payouts] ${label} backend call failed`, err);
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
    fireBackendPayoutAction("approve", apiApprovePayout(payoutId));
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
    fireBackendPayoutAction("hold", apiHoldPayout(payoutId, reason));
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
    fireBackendPayoutAction(
      "markPaid",
      apiMarkPayoutPaid(payoutId, reference),
    );
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
    fireBackendPayoutAction(
      "bulkApprove",
      apiBulkApprovePayouts(payoutIds),
    );
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
    fireBackendPayoutAction(
      "bulkHold",
      apiBulkHoldPayouts(payoutIds, reason),
    );
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
