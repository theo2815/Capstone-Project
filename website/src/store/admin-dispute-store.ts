import { create } from "zustand";
import {
  ADMIN_DISPUTES,
  type Dispute,
  type DisputeOrderSnapshot,
  type DisputePhotoSnapshot,
  type DisputeReason,
  type DisputeResolution,
} from "@/lib/admin-disputes";

// Mock-only store of admin actions on the disputes queue. NOT persisted —
// real backend ships in Phase F. Sparse `overrides` map keyed by dispute
// id; merged view reconstructed per render. Decision log is append-only
// (capped at 50) so the dispute Activity slab stays bounded.
//
// Runner-submitted disputes (post-2026-05-08): submitDispute() pushes a fully-
// formed Dispute into `submissions[]`. getEffectiveDisputes() concatenates
// seed + submissions, then layers admin overrides on top. Submissions are NOT
// keyed in `overrides` because they aren't patches — they're net-new disputes.

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
  ) => void;
  deny: (disputeId: string, reason: string | null) => void;
  escalate: (disputeId: string, reason: string | null) => void;
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
  resolve: (disputeId, { resolution, refundAmount, reason }) =>
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
    })),
  deny: (disputeId, reason) =>
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
    })),
  escalate: (disputeId, reason) =>
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
    })),
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
