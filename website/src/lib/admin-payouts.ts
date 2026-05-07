import {
  PAYOUT_METHOD_LABEL,
  type PayoutMethod,
} from "@/store/photographer-settings-store";

// Phase 2b — admin payout review queue. Distinct from the photographer-
// facing PhotographerPayout (which is the photographer's view of their own
// cycles in `lib/photographer-mock.ts`). Admin sees a flat list of every
// photographer's pending cycles with status transitions:
//   pending_review → approved → paid
//   pending_review ↔ held (any state can move to held except paid)
// Bulk actions wrap multiple cycle IDs into a single mutation.
//
// Payout accounts are SNAPSHOTTED onto each cycle at submission time. The
// admin transfer is a manual operation — admin reads `payoutAccount` from
// the cycle, sends the money via their banking app, then marks the row
// Paid. Snapshotting (rather than live-looking-up the photographer's
// current primary account) means an account-info update by the photographer
// after submission does NOT silently re-route an in-flight transfer.

export type AdminPayoutStatus = "pending_review" | "approved" | "held" | "paid";

export type PayoutAccountSnapshot = {
  method: PayoutMethod;
  /** Raw digits — rendered in full so admin can copy into their banking app. */
  accountNumber: string;
  accountName: string;
  /**
   * Photographer-uploaded QR. Mirrors the per-account QR in
   * photographer-settings-store; null when the photographer hasn't uploaded
   * one yet. Real backend (Phase F) stores the dataUrl privately and only
   * surfaces it to admin scopes.
   */
  qr: { dataUrl: string; uploadedAt: string } | null;
};

export interface AdminPayoutCycle {
  id: string;
  photographerId: string;
  photographerName: string;
  brandName: string | null;
  handle: string | null;
  weekOf: string; // ISO Monday
  amount: number; // peso amount kept by photographer this cycle
  itemCount: number; // sales count
  method: PayoutMethod;
  status: AdminPayoutStatus;
  submittedAt: string;
  reviewedAt: string | null;
  paidAt: string | null;
  paymentReference: string | null;
  holdReason: string | null;
  payoutAccount: PayoutAccountSnapshot;
}

export const ADMIN_PAYOUT_STATUS_LABEL: Record<AdminPayoutStatus, string> = {
  pending_review: "Pending review",
  approved: "Approved",
  held: "Held",
  paid: "Paid",
};

export function payoutMethodLabel(method: PayoutMethod): string {
  return PAYOUT_METHOD_LABEL[method];
}

// Per-photographer account snapshots used to seed every cycle. Realistic PH
// mobile numbers (placeholder digits — not real). Admin sees these as the
// "TRANSFER TO" reference card in the Mark-paid modal + drawer; the modal
// also surfaces the QR (downloadable) for admins who prefer QR-scan transfers.
const PAKSIT_GCASH: PayoutAccountSnapshot = {
  method: "gcash",
  accountNumber: "09175504523",
  accountName: "Maricel Casinto",
  qr: {
    dataUrl: "mock://qr/paksit-gcash",
    uploadedAt: "2026-04-01T08:30:00.000Z",
  },
};
const CEBUSTRIDE_MAYA: PayoutAccountSnapshot = {
  method: "maya",
  accountNumber: "09286612847",
  accountName: "Brian Yap",
  qr: {
    dataUrl: "mock://qr/cebustride-maya",
    uploadedAt: "2026-04-02T11:10:00.000Z",
  },
};

export const ADMIN_PAYOUT_SEED: ReadonlyArray<AdminPayoutCycle> = [
  // 4 pending_review
  {
    id: "PAY-2026W18-PAKSIT",
    photographerId: "photog-paksit",
    photographerName: "Paksit Studio",
    brandName: "Paksit Photos",
    handle: "paksitphotos",
    weekOf: "2026-04-27",
    amount: 8420,
    itemCount: 56,
    method: "gcash",
    status: "pending_review",
    submittedAt: "2026-05-04T01:00:00.000Z",
    reviewedAt: null,
    paidAt: null,
    paymentReference: null,
    holdReason: null,
    payoutAccount: PAKSIT_GCASH,
  },
  {
    id: "PAY-2026W18-CEBUSTRIDE",
    photographerId: "photog-cebustride",
    photographerName: "Cebu Stride",
    brandName: "Cebu Stride",
    handle: "cebustride",
    weekOf: "2026-04-27",
    amount: 6240,
    itemCount: 41,
    method: "maya",
    status: "pending_review",
    submittedAt: "2026-05-04T01:00:00.000Z",
    reviewedAt: null,
    paidAt: null,
    paymentReference: null,
    holdReason: null,
    payoutAccount: CEBUSTRIDE_MAYA,
  },
  {
    id: "PAY-2026W17-PAKSIT",
    photographerId: "photog-paksit",
    photographerName: "Paksit Studio",
    brandName: "Paksit Photos",
    handle: "paksitphotos",
    weekOf: "2026-04-20",
    amount: 7180,
    itemCount: 48,
    method: "gcash",
    status: "pending_review",
    submittedAt: "2026-04-27T01:00:00.000Z",
    reviewedAt: null,
    paidAt: null,
    paymentReference: null,
    holdReason: null,
    payoutAccount: PAKSIT_GCASH,
  },
  {
    id: "PAY-2026W17-CEBUSTRIDE",
    photographerId: "photog-cebustride",
    photographerName: "Cebu Stride",
    brandName: "Cebu Stride",
    handle: "cebustride",
    weekOf: "2026-04-20",
    amount: 5320,
    itemCount: 35,
    method: "maya",
    status: "pending_review",
    submittedAt: "2026-04-27T01:00:00.000Z",
    reviewedAt: null,
    paidAt: null,
    paymentReference: null,
    holdReason: null,
    payoutAccount: CEBUSTRIDE_MAYA,
  },

  // 2 approved (waiting for mark-paid)
  {
    id: "PAY-2026W16-PAKSIT",
    photographerId: "photog-paksit",
    photographerName: "Paksit Studio",
    brandName: "Paksit Photos",
    handle: "paksitphotos",
    weekOf: "2026-04-13",
    amount: 9120,
    itemCount: 62,
    method: "gcash",
    status: "approved",
    submittedAt: "2026-04-20T01:00:00.000Z",
    reviewedAt: "2026-04-21T09:30:00.000Z",
    paidAt: null,
    paymentReference: null,
    holdReason: null,
    payoutAccount: PAKSIT_GCASH,
  },
  {
    id: "PAY-2026W16-CEBUSTRIDE",
    photographerId: "photog-cebustride",
    photographerName: "Cebu Stride",
    brandName: "Cebu Stride",
    handle: "cebustride",
    weekOf: "2026-04-13",
    amount: 4810,
    itemCount: 32,
    method: "maya",
    status: "approved",
    submittedAt: "2026-04-20T01:00:00.000Z",
    reviewedAt: "2026-04-21T09:32:00.000Z",
    paidAt: null,
    paymentReference: null,
    holdReason: null,
    payoutAccount: CEBUSTRIDE_MAYA,
  },

  // 1 held (chargeback risk)
  {
    id: "PAY-2026W15-CEBUSTRIDE",
    photographerId: "photog-cebustride",
    photographerName: "Cebu Stride",
    brandName: "Cebu Stride",
    handle: "cebustride",
    weekOf: "2026-04-06",
    amount: 5680,
    itemCount: 38,
    method: "maya",
    status: "held",
    submittedAt: "2026-04-13T01:00:00.000Z",
    reviewedAt: "2026-04-14T11:15:00.000Z",
    paidAt: null,
    paymentReference: null,
    holdReason: "Chargeback rate elevated this week — pending payments review.",
    payoutAccount: CEBUSTRIDE_MAYA,
  },

  // 3 paid (historical)
  {
    id: "PAY-2026W15-PAKSIT",
    photographerId: "photog-paksit",
    photographerName: "Paksit Studio",
    brandName: "Paksit Photos",
    handle: "paksitphotos",
    weekOf: "2026-04-06",
    amount: 8260,
    itemCount: 54,
    method: "gcash",
    status: "paid",
    submittedAt: "2026-04-13T01:00:00.000Z",
    reviewedAt: "2026-04-13T14:00:00.000Z",
    paidAt: "2026-04-15T10:00:00.000Z",
    paymentReference: "GC-77B221",
    holdReason: null,
    payoutAccount: PAKSIT_GCASH,
  },
  {
    id: "PAY-2026W14-PAKSIT",
    photographerId: "photog-paksit",
    photographerName: "Paksit Studio",
    brandName: "Paksit Photos",
    handle: "paksitphotos",
    weekOf: "2026-03-30",
    amount: 7440,
    itemCount: 49,
    method: "gcash",
    status: "paid",
    submittedAt: "2026-04-06T01:00:00.000Z",
    reviewedAt: "2026-04-06T13:00:00.000Z",
    paidAt: "2026-04-08T10:00:00.000Z",
    paymentReference: "GC-58A114",
    holdReason: null,
    payoutAccount: PAKSIT_GCASH,
  },
  {
    id: "PAY-2026W14-CEBUSTRIDE",
    photographerId: "photog-cebustride",
    photographerName: "Cebu Stride",
    brandName: "Cebu Stride",
    handle: "cebustride",
    weekOf: "2026-03-30",
    amount: 4920,
    itemCount: 33,
    method: "maya",
    status: "paid",
    submittedAt: "2026-04-06T01:00:00.000Z",
    reviewedAt: "2026-04-06T13:02:00.000Z",
    paidAt: "2026-04-08T10:01:00.000Z",
    paymentReference: "MY-32C887",
    holdReason: null,
    payoutAccount: CEBUSTRIDE_MAYA,
  },
];

export function getPendingPayouts(): AdminPayoutCycle[] {
  return ADMIN_PAYOUT_SEED.filter((p) => p.status === "pending_review");
}

export function getPayoutById(id: string): AdminPayoutCycle | undefined {
  return ADMIN_PAYOUT_SEED.find((p) => p.id === id);
}
