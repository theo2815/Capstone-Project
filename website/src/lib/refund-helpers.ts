// Pure helpers for runner-side refund flow. Operate on the `RunnerDispute[]`
// embedded on every order payload from /me/orders — no store reads, no
// side effects.
//
// Shape decisions:
//  - One dispute per photo (matches BE `Dispute.photoId` singular shape).
//    A multi-photo refund request creates N disputes sharing reason+note.
//  - "kind" enum on the receipt status chip mirrors what runner sees:
//      none      — no disputes for this order (or every dispute withdrawn)
//      pending   — every disputed photo still open / escalated
//      partial   — some photos disputed/resolved, others untouched
//      approved  — every disputed photo has been resolved with a refund
//      rejected  — every disputed photo was denied with no approvals
//  - Withdrawn disputes are filtered out of every rollup — they're the
//    runner's own cancellation and shouldn't carry visual weight or block
//    re-submission. Denied disputes are also re-submittable but DO surface
//    a "Refund declined" chip with the admin note.

import type { RunnerDispute } from "@/lib/api-orders";
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

export function getOrderRefundStatus(
  order: MockOrder,
): OrderRefundStatus {
  const visible = (order.disputes ?? []).filter(
    (d) => d.status !== "withdrawn",
  );
  const photoCount = order.photoIds.length;

  const pendingCount = visible.filter(
    (d) => d.status === "open" || d.status === "escalated",
  ).length;
  const approved = visible.filter(
    (d) => d.status === "resolved" && d.refundAmount !== null,
  );
  const approvedCount = approved.length;
  const refundAmount = approved.reduce(
    (sum, d) => sum + (d.refundAmount ?? 0),
    0,
  );
  const rejected = visible.filter((d) => d.status === "denied");
  const rejectedCount = rejected.length;
  // Prefer the admin's resolution note over the runner's own submission note
  // — when the chip says "Refund declined" the runner cares why admin said no.
  const rejectedNote = rejected[0]?.resolutionNote ?? rejected[0]?.note ?? null;
  const totalDisputed = visible.length;

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
// already attached to an open / escalated / resolved dispute. Denied and
// withdrawn disputes leave the photo eligible for re-submission.
export function getDisputableePhotoIds(order: MockOrder): string[] {
  const blocked = new Set(
    (order.disputes ?? [])
      .filter((d) => d.status !== "denied" && d.status !== "withdrawn")
      .map((d) => d.photoId),
  );
  return order.photoIds.filter((id) => !blocked.has(id));
}

// Whether the runner can submit any new dispute for this order. False once
// every photo is locked into an open / escalated / approved dispute.
export function canSubmitRefund(order: MockOrder): boolean {
  return getDisputableePhotoIds(order).length > 0;
}
