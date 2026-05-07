// Pure helpers for runner-side refund flow. No side effects, no store reads —
// pass disputes in. Consumers wire to useAdminDisputeStore + getEffectiveDisputes.
//
// Shape decisions:
//  - One dispute per photo (matches existing Dispute.photoId singular shape).
//    A multi-photo refund request creates N disputes sharing reason+note.
//  - "kind" enum on the receipt status chip mirrors what runner sees:
//      none      — no disputes for this order
//      pending   — every disputed photo still open, no resolution yet
//      partial   — some photos disputed/resolved, others untouched
//      approved  — every disputed photo has been resolved with a refund
//      rejected  — every disputed photo was denied with no approvals

import type { Dispute } from "@/lib/admin-disputes";
import type { MockOrder } from "@/store/orders-store";

export type RefundStatusKind =
  | "none"
  | "pending"
  | "partial"
  | "approved"
  | "rejected";

export interface OrderRefundStatus {
  kind: RefundStatusKind;
  refundAmount: number;
  pendingCount: number;
  approvedCount: number;
  rejectedCount: number;
  rejectedNote: string | null;
  totalDisputed: number;
}

export function getDisputesForOrder(
  orderId: string,
  disputes: ReadonlyArray<Dispute>,
): Dispute[] {
  return disputes.filter((d) => d.orderId === orderId);
}

export function getOrderRefundStatus(
  order: MockOrder,
  disputes: ReadonlyArray<Dispute>,
): OrderRefundStatus {
  const orderDisputes = getDisputesForOrder(order.id, disputes);
  const photoCount = order.photoIds.length;

  const pendingCount = orderDisputes.filter(
    (d) => d.status === "open" || d.status === "escalated",
  ).length;
  const approved = orderDisputes.filter(
    (d) => d.status === "resolved" && d.refundAmount !== null,
  );
  const approvedCount = approved.length;
  const refundAmount = approved.reduce(
    (sum, d) => sum + (d.refundAmount ?? 0),
    0,
  );
  const rejected = orderDisputes.filter((d) => d.status === "denied");
  const rejectedCount = rejected.length;
  const rejectedNote = rejected[0]?.note ?? null;
  const totalDisputed = orderDisputes.length;

  let kind: RefundStatusKind = "none";
  if (totalDisputed === 0) {
    kind = "none";
  } else if (pendingCount > 0) {
    kind = totalDisputed < photoCount ? "partial" : "pending";
  } else if (approvedCount > 0 && rejectedCount === 0) {
    kind = "approved";
  } else if (rejectedCount > 0 && approvedCount === 0) {
    kind = "rejected";
  } else {
    kind = "partial";
  }

  return {
    kind,
    refundAmount,
    pendingCount,
    approvedCount,
    rejectedCount,
    rejectedNote,
    totalDisputed,
  };
}

// Photo IDs the runner can still dispute for this order. Excludes any photo
// already attached to an open / escalated / resolved dispute (a denied
// dispute can be re-submitted — the runner has new evidence).
export function getDisputableePhotoIds(
  order: MockOrder,
  disputes: ReadonlyArray<Dispute>,
): string[] {
  const blocked = new Set(
    getDisputesForOrder(order.id, disputes)
      .filter((d) => d.status !== "denied")
      .map((d) => d.photoId),
  );
  return order.photoIds.filter((id) => !blocked.has(id));
}

// Whether the runner can submit any new dispute for this order. False once
// every photo is locked into an open / approved dispute.
export function canSubmitRefund(
  order: MockOrder,
  disputes: ReadonlyArray<Dispute>,
): boolean {
  return getDisputableePhotoIds(order, disputes).length > 0;
}
