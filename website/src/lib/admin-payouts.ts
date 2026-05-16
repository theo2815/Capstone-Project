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

export const ADMIN_PAYOUT_SEED: ReadonlyArray<AdminPayoutCycle> = [];

export function getPendingPayouts(): AdminPayoutCycle[] {
  return ADMIN_PAYOUT_SEED.filter((p) => p.status === "pending_review");
}

export function getPayoutById(id: string): AdminPayoutCycle | undefined {
  return ADMIN_PAYOUT_SEED.find((p) => p.id === id);
}
