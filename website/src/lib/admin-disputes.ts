export type DisputeStatus = "open" | "resolved" | "denied" | "escalated" | "withdrawn";
// Server-derived "what this dispute needs now" (backend DisputeAttention wire values).
export type DisputeAttention =
  | "review"
  | "escalated"
  | "send_link"
  | "awaiting_claim"
  | "link_expired"
  | "check_paymongo"
  | "refund_problem"
  | "refund_in_progress"
  | "closed";
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
export interface DisputeActivityEntry {
  id: string;
  decidedAt: string;
  decision:
    | "resolved"
    | "denied"
    | "escalated"
    | "refund_sent_to_manual_review"
    | "refund_link_sent"
    | "refund_cancelled";
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
  // neither order nor photo can resolve the event (rare). Older responses
  // may omit it. Prefer it over a local-catalog lookup.
  eventName?: string | null;
  runnerHandle: string;
  photographerHandle: string;
  reason: DisputeReason;
  note: string;
  status: DisputeStatus;
  reportedAt: string;
  resolvedAt: string | null;
  refundAmount: number | null;
  refundStatus?: string | null;
  // The settled PayMongo payment (pay_ id, and its cs_/pi_ reference) to find it in the PayMongo dashboard.
  paymentId?: string | null;
  paymentRef?: string | null;
  // Manual QR Ph refund: the transfer link sent to the runner, and when.
  refundLinkUrl?: string | null;
  refundLinkSentAt?: string | null;
  refundLinkExpiresAt?: string | null;
  // Parked QR Ph refund only: when PayMongo stops refunding, and when an unfinished refund is cancelled.
  refundWindowClosesAt?: string | null;
  refundCancelsAt?: string | null;
  itemChargeAmount?: number | null;
  resolution: DisputeResolution | null;
  orderSnapshot: DisputeOrderSnapshot;
  photoSnapshot: DisputePhotoSnapshot;
  // Server-side audit trail — every admin action ever taken on this
  // dispute, newest first. Persisted by the backend across sessions.
  activity?: DisputeActivityEntry[];
  // Due times worked out by the server on read (DW-1); older responses omit them.
  refundRequestedAt?: string | null;
  attention?: DisputeAttention;
  dueAt?: string | null;
  overdue?: boolean;
  linkExpiringSoon?: boolean;
  // Can a full refund be approved now (DW-3)? Wire codes match the deny reason codes bought_together / window_closed.
  refundable?: boolean | null;
  refundBlockedReason?: "bought_together" | "window_closed" | "no_payment" | null;
  refundBlockedMessage?: string | null;
  paymentPhotoCount?: number | null;
}

// One row of GET /admin/disputes/deny-reasons; the server owns the list and the runner text.
export interface DisputeDenyReason {
  code: string;
  label: string;
  runnerText: string;
  noteRequired: boolean;
}

export const DISPUTE_STATUS_LABEL: Record<DisputeStatus, string> = {
  open: "Open",
  resolved: "Resolved",
  denied: "Denied",
  escalated: "Escalated",
  withdrawn: "Withdrawn",
};

// Values match AdminStatusPillTone; lib stays free of component imports.
export function disputeStatusTone(status: DisputeStatus): "amber" | "fresh" | "ink" | "slate" | "muted" {
  switch (status) {
    case "open":
      return "amber";
    case "resolved":
      return "fresh";
    case "denied":
      return "ink";
    case "escalated":
      return "slate";
    case "withdrawn":
      return "muted";
  }
}

// What the admin must do next; "closed" shows no chip.
export const DISPUTE_ATTENTION_LABEL: Record<DisputeAttention, string | null> = {
  review: "Needs review",
  escalated: "Escalated: decide",
  send_link: "Approved: send the refund link",
  awaiting_claim: "Link sent: waiting for the runner",
  link_expired: "Link expired: send a new one",
  check_paymongo: "Check PayMongo",
  refund_problem: "Refund problem: check PayMongo",
  refund_in_progress: "Refund in progress",
  closed: null,
};

export const DISPUTE_PAYMENT_LABEL: Record<string, string> = {
  qrph: "QR Ph",
  gcash: "GCash",
  maya: "Maya",
  paymaya: "PayMaya",
  card: "Card",
  grabpay: "GrabPay",
};

export const DISPUTE_REASON_LABEL: Record<DisputeReason, string> = {
  wrong_runner: "Wrong runner in photo",
  low_quality: "Too blurred to recognise",
  not_received: "Won't download or file is broken",
  duplicate_charge: "Charged more than once",
  other: "Checkout error or something else",
};

export const DISPUTE_RESOLUTION_LABEL: Record<DisputeResolution, string> = {
  refund_full: "Full refund",
  refund_partial: "Partial refund",
  deny: "Claim denied",
};
