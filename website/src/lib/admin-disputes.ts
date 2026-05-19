// Phase 2b — populated. The full Dispute shape carries inline order +
// photo snapshots so the dispute UI never reaches into orders-store or
// photo registries. When backend wiring lands in Phase F, the snapshots
// stay (they're write-once at dispute creation time on the server).

export type DisputeStatus = "open" | "resolved" | "denied" | "escalated";
export type DisputeReason =
  | "wrong_runner"
  | "low_quality"
  | "not_received"
  | "duplicate_charge"
  | "other";
export type DisputeResolution = "refund_full" | "refund_partial" | "deny";

export interface DisputeOrderSnapshot {
  total: number;
  paymentMethod: string;
  paidAt: string;
}

export interface DisputePhotoSnapshot {
  alt: string;
  kmMark: number | null;
  bib: string | null;
  thumbnailUrl?: string;
}

// Mirrors BE DisputeActivityEntry. One row per admin_decision_log entry
// targeting this dispute. resolution + refundAmount are flattened from the
// BE meta JSONB (only present on decision="resolved" entries).
// `decision` matches the union in store/admin-dispute-store.ts so the local
// optimistic log and the BE-persisted activity render through the same
// label table.
export interface DisputeActivityEntry {
  id: string;
  decidedAt: string;
  decision: "resolved" | "denied" | "escalated";
  resolution: DisputeResolution | null;
  refundAmount: number | null;
  reason: string | null;
}

export interface Dispute {
  id: string;
  orderId: string;
  photoId: string;
  eventId: string;
  // Hydrated from BE — the actual event title. Null for ghost rows where
  // neither order nor photo can resolve the event (rare). Optional so
  // local optimistic mints (`useAdminDisputeStore.submitDispute`) don't
  // need to fabricate it. Always prefer this over a local-catalog lookup,
  // which is empty post-Phase B.
  eventName?: string | null;
  runnerHandle: string;
  photographerHandle: string;
  reason: DisputeReason;
  note: string;
  status: DisputeStatus;
  reportedAt: string;
  resolvedAt: string | null;
  refundAmount: number | null;
  resolution: DisputeResolution | null;
  orderSnapshot: DisputeOrderSnapshot;
  photoSnapshot: DisputePhotoSnapshot;
  // Server-side audit trail — every admin action ever taken on this
  // dispute, newest first. Persists across sessions (unlike the
  // session-only optimistic log in useAdminDisputeStore.log).
  activity?: DisputeActivityEntry[];
}

export const DISPUTE_REASON_LABEL: Record<DisputeReason, string> = {
  wrong_runner: "Wrong runner in photo",
  low_quality: "Photo quality too low",
  not_received: "Order paid but never delivered",
  duplicate_charge: "Charged twice for same order",
  other: "Other (see note)",
};

export const DISPUTE_RESOLUTION_LABEL: Record<DisputeResolution, string> = {
  refund_full: "Full refund",
  refund_partial: "Partial refund",
  deny: "Claim denied",
};

export const ADMIN_DISPUTES: ReadonlyArray<Dispute> = [];

export function getOpenDisputes(): Dispute[] {
  return ADMIN_DISPUTES.filter((d) => d.status === "open");
}

export function getDisputeById(id: string): Dispute | undefined {
  return ADMIN_DISPUTES.find((d) => d.id === id);
}
