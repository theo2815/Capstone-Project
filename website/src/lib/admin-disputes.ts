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

export const ADMIN_DISPUTES: ReadonlyArray<Dispute> = [
  // 3 open
  {
    id: "DSP-A1B2C3",
    orderId: "QP-77F2A1",
    photoId: "photo-bay-run-228",
    eventId: "1",
    runnerHandle: "juan",
    photographerHandle: "paksitphotos",
    reason: "wrong_runner",
    note: "The bib on this photo is mine but the runner in the frame is wearing a different jersey. I think the AI matched on the bib digits but the photo is of someone else with the same bib reuse.",
    status: "open",
    reportedAt: "2026-05-05T09:14:00.000Z",
    resolvedAt: null,
    refundAmount: null,
    resolution: null,
    orderSnapshot: {
      total: 180,
      paymentMethod: "gcash",
      paidAt: "2026-04-28T18:42:00.000Z",
    },
    photoSnapshot: {
      alt: "Runner crossing the SRP boulevard mid-stride",
      kmMark: 21,
      bib: "1247",
      thumbnailUrl: undefined,
    },
  },
  {
    id: "DSP-D4E5F6",
    orderId: "QP-91A8C2",
    photoId: "photo-stride-074",
    eventId: "u3",
    runnerHandle: "thea",
    photographerHandle: "cebustride",
    reason: "low_quality",
    note: "Photo is heavily blurred, almost unusable. I expected the AI cull to catch this. Asking for a refund or a different photo from the same kilometer mark.",
    status: "open",
    reportedAt: "2026-05-06T13:02:00.000Z",
    resolvedAt: null,
    refundAmount: null,
    resolution: null,
    orderSnapshot: {
      total: 120,
      paymentMethod: "maya",
      paidAt: "2026-05-04T11:18:00.000Z",
    },
    photoSnapshot: {
      alt: "Runner approaching aid station — partial blur on subject",
      kmMark: 12,
      bib: "0863",
      thumbnailUrl: undefined,
    },
  },
  {
    id: "DSP-G7H8I9",
    orderId: "QP-15D9B7",
    photoId: "photo-sunrise-5k-018",
    eventId: "u2",
    runnerHandle: "nico",
    photographerHandle: "paksitphotos",
    reason: "not_received",
    note: "Paid for the bundle 6 hours ago. The photo never appeared in my orders. The download link in the email is broken.",
    status: "open",
    reportedAt: "2026-05-07T01:30:00.000Z",
    resolvedAt: null,
    refundAmount: null,
    resolution: null,
    orderSnapshot: {
      total: 350,
      paymentMethod: "gcash",
      paidAt: "2026-05-06T19:42:00.000Z",
    },
    photoSnapshot: {
      alt: "Runner at sunrise warm-up zone",
      kmMark: null,
      bib: "0421",
      thumbnailUrl: undefined,
    },
  },

  // 1 resolved (partial refund)
  {
    id: "DSP-J1K2L3",
    orderId: "QP-44E6F8",
    photoId: "photo-stride-099",
    eventId: "1",
    runnerHandle: "maria",
    photographerHandle: "cebustride",
    reason: "duplicate_charge",
    note: "I see two ₱150 charges on my GCash for the same order. Refund one of them please.",
    status: "resolved",
    reportedAt: "2026-04-15T08:11:00.000Z",
    resolvedAt: "2026-04-16T14:33:00.000Z",
    refundAmount: 150,
    resolution: "refund_partial",
    orderSnapshot: {
      total: 300,
      paymentMethod: "gcash",
      paidAt: "2026-04-14T20:01:00.000Z",
    },
    photoSnapshot: {
      alt: "Finish-line runner with both arms raised",
      kmMark: 42,
      bib: "2089",
      thumbnailUrl: undefined,
    },
  },

  // 1 denied
  {
    id: "DSP-M4N5O6",
    orderId: "QP-22A4D1",
    photoId: "photo-bay-run-411",
    eventId: "1",
    runnerHandle: "jp",
    photographerHandle: "paksitphotos",
    reason: "low_quality",
    note: "Photo is a bit dark, want a refund.",
    status: "denied",
    reportedAt: "2026-04-22T15:50:00.000Z",
    resolvedAt: "2026-04-23T10:12:00.000Z",
    refundAmount: null,
    resolution: "deny",
    orderSnapshot: {
      total: 180,
      paymentMethod: "gcash",
      paidAt: "2026-04-21T17:00:00.000Z",
    },
    photoSnapshot: {
      alt: "Runner mid-stride at golden hour",
      kmMark: 18,
      bib: "0517",
      thumbnailUrl: undefined,
    },
  },

  // 1 escalated
  {
    id: "DSP-P7Q8R9",
    orderId: "QP-58C9E3",
    photoId: "photo-stride-156",
    eventId: "u3",
    runnerHandle: "thea",
    photographerHandle: "cebustride",
    reason: "other",
    note: "Photo shows my face but I never registered for this race. I think someone uploaded a photo of me from a public area without consent. Asking for full refund AND removal.",
    status: "escalated",
    reportedAt: "2026-04-29T12:08:00.000Z",
    resolvedAt: null,
    refundAmount: null,
    resolution: null,
    orderSnapshot: {
      total: 250,
      paymentMethod: "gcash",
      paidAt: "2026-04-28T16:34:00.000Z",
    },
    photoSnapshot: {
      alt: "Spectator near finish line — disputed identity match",
      kmMark: null,
      bib: null,
      thumbnailUrl: undefined,
    },
  },
];

export function getOpenDisputes(): Dispute[] {
  return ADMIN_DISPUTES.filter((d) => d.status === "open");
}

export function getDisputeById(id: string): Dispute | undefined {
  return ADMIN_DISPUTES.find((d) => d.id === id);
}
