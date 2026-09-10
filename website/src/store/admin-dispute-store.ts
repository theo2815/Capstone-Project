import { create } from "zustand";
import { useToastStore } from "@/store/toast-store";
import {
  ADMIN_DISPUTES,
  type Dispute,
  type DisputeOrderSnapshot,
  type DisputePhotoSnapshot,
  type DisputeReason,
  type DisputeResolution,
} from "@/lib/admin-disputes";
import {
  resolveDispute as apiResolveDispute,
  denyDispute as apiDenyDispute,
  escalateDispute as apiEscalateDispute,
} from "@/lib/api-admin";
// Photographer in-app notifications are pushed BE-side by
// AdminDisputeService — see AdminDecisionLogService.pushMessage calls in
// resolve / deny / escalate. The FE reads them via
// GET /me/photographer/messages and no longer fires FE-side notify helpers.

// Admin decisions are mirrored locally only after backend confirmation.
// The backend also sends photographer notifications in the same transaction.
//
// Runner-submitted disputes (post-2026-05-08): submitDispute() pushes a fully-
// formed Dispute into `submissions[]`. getEffectiveDisputes() concatenates
// seed + submissions, then layers admin overrides on top. Submissions are NOT
// keyed in `overrides` because they aren't patches — they're net-new disputes.
// Runner-side write travels via `submitOrderRefund` in `api-orders.ts` per
// Phase E wiring; the admin store mirrors locally for the demo.

function toastBackendFailure(label: string, error: unknown): void {
  console.error(`[admin/disputes] ${label} failed`, error);
  useToastStore.getState().showToast({
    kind: "error",
    message: `Couldn't ${label} dispute — please try again.`,
  });
}

export type DisputeDecision = "resolved" | "denied" | "escalated";

export interface DisputeLogEntry {
  disputeId: string;
  decision: DisputeDecision;
  resolution: DisputeResolution | null;
  refundAmount: number | null;
  reason: string | null;
  decidedAt: string;
}

export interface SubmitDisputePayload {
  orderId: string;
  photoId: string;
  eventId: string;
  runnerHandle: string;
  photographerHandle: string;
  reason: DisputeReason;
  note: string;
  orderSnapshot: DisputeOrderSnapshot;
  photoSnapshot: DisputePhotoSnapshot;
}

interface AdminDisputeStoreState {
  overrides: Record<string, Partial<Dispute>>;
  submissions: Dispute[];
  log: DisputeLogEntry[];
  resolve: (
    disputeId: string,
    args: {
      resolution: DisputeResolution;
      refundAmount: number | null;
      reason: string | null;
    },
  ) => Promise<boolean>;
  deny: (disputeId: string, reason: string | null) => Promise<boolean>;
  escalate: (disputeId: string, reason: string | null) => Promise<boolean>;
  submitDispute: (payload: SubmitDisputePayload) => Dispute;
  clear: () => void;
}

// Mock-only id generator. Backend will assign in Phase F.
function makeRunnerDisputeId(): string {
  const slug = Math.random().toString(36).slice(2, 8).toUpperCase();
  return `DSP-RUN-${slug}`;
}

function appendLog(
  prev: DisputeLogEntry[],
  entry: DisputeLogEntry,
): DisputeLogEntry[] {
  return [entry, ...prev].slice(0, 50);
}

export const useAdminDisputeStore = create<AdminDisputeStoreState>((set) => ({
  overrides: {},
  submissions: [],
  log: [],
  resolve: async (disputeId, { resolution, refundAmount, reason }) => {
    try {
      await apiResolveDispute(disputeId, { resolution, refundAmount, reason });
      set((s) => ({
        overrides: {
          ...s.overrides,
          [disputeId]: {
            ...s.overrides[disputeId],
            status: "resolved",
            resolution,
            refundAmount,
            resolvedAt: new Date().toISOString(),
          },
        },
        log: appendLog(s.log, {
          disputeId,
          decision: "resolved",
          resolution,
          refundAmount,
          reason,
          decidedAt: new Date().toISOString(),
        }),
      }));
      return true;
    } catch (error) {
      toastBackendFailure("resolve", error);
      return false;
    }
  },
  deny: async (disputeId, reason) => {
    try {
      await apiDenyDispute(disputeId, reason);
      set((s) => ({
        overrides: {
          ...s.overrides,
          [disputeId]: {
            ...s.overrides[disputeId],
            status: "denied",
            resolution: "deny",
            refundAmount: null,
            resolvedAt: new Date().toISOString(),
          },
        },
        log: appendLog(s.log, {
          disputeId,
          decision: "denied",
          resolution: "deny",
          refundAmount: null,
          reason,
          decidedAt: new Date().toISOString(),
        }),
      }));
      return true;
    } catch (error) {
      toastBackendFailure("deny", error);
      return false;
    }
  },
  escalate: async (disputeId, reason) => {
    try {
      await apiEscalateDispute(disputeId, reason);
      set((s) => ({
        overrides: {
          ...s.overrides,
          [disputeId]: {
            ...s.overrides[disputeId],
            status: "escalated",
          },
        },
        log: appendLog(s.log, {
          disputeId,
          decision: "escalated",
          resolution: null,
          refundAmount: null,
          reason,
          decidedAt: new Date().toISOString(),
        }),
      }));
      return true;
    } catch (error) {
      toastBackendFailure("escalate", error);
      return false;
    }
  },
  submitDispute: (payload) => {
    const dispute: Dispute = {
      id: makeRunnerDisputeId(),
      orderId: payload.orderId,
      photoId: payload.photoId,
      eventId: payload.eventId,
      runnerHandle: payload.runnerHandle,
      photographerHandle: payload.photographerHandle,
      reason: payload.reason,
      note: payload.note,
      status: "open",
      reportedAt: new Date().toISOString(),
      resolvedAt: null,
      refundAmount: null,
      resolution: null,
      orderSnapshot: payload.orderSnapshot,
      photoSnapshot: payload.photoSnapshot,
    };
    set((s) => ({ submissions: [dispute, ...s.submissions] }));
    return dispute;
  },
  clear: () => set({ overrides: {}, submissions: [], log: [] }),
}));

export function mergeDispute(
  base: Dispute,
  patch: Partial<Dispute> | undefined,
): Dispute {
  return patch ? { ...base, ...patch } : base;
}

export function getEffectiveDisputes(
  overrides: Record<string, Partial<Dispute>>,
  submissions: ReadonlyArray<Dispute> = [],
): Dispute[] {
  const seed = ADMIN_DISPUTES.map((d) => mergeDispute(d, overrides[d.id]));
  const submitted = submissions.map((d) => mergeDispute(d, overrides[d.id]));
  return [...submitted, ...seed];
}

/**
 * Merges live server data with local overrides + runner-submitted disputes
 * for instant feedback after admin actions. Use this in components that
 * call `useAdminDisputes()` directly.
 */
export function mergeDisputesWithOverrides(
  serverData: ReadonlyArray<Dispute>,
  overrides: Record<string, Partial<Dispute>>,
  submissions: ReadonlyArray<Dispute> = [],
): Dispute[] {
  const server = serverData.map((d) => mergeDispute(d, overrides[d.id]));
  const submitted = submissions.map((d) => mergeDispute(d, overrides[d.id]));
  return [...submitted, ...server];
}
