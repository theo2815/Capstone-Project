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

export interface Dispute {
  id: string;
  orderId: string;
  photoId: string;
  eventId: string;
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
