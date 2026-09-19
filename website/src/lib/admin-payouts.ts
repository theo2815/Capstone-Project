import {
  PAYOUT_METHOD_LABEL,
  type PayoutMethod,
} from "@/store/photographer-settings-store";

// Admin payout review queue. Distinct from the photographer-
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

// Admin wire vocabulary (AdminPayoutService.toAdminWireStatus). pending_review →
// approved → processing → paid is the happy path; held pauses; reject/failed/
// returned/cancelled are terminal-with-a-reason. failed = money never left
// (safe), returned = money left and bounced back.
export type AdminPayoutStatus =
  | "pending_review"
  | "approved"
  | "processing"
  | "held"
  | "paid"
  | "failed"
  | "returned"
  | "rejected"
  | "cancelled";

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
  provider?: string;
  submittedAt: string;
  reviewedAt: string | null;
  paidAt: string | null;
  paymentReference: string | null;
  holdReason: string | null;
  /** Admin's reason when rejected / failed; null otherwise. */
  rejectedReason: string | null;
  failedReason: string | null;
  /**
   * Two-person approval (large cycles). approvedBy is the first approver;
   * secondApprovedBy the distinct second. Both null until someone approves;
   * a large cycle sits at "1 of 2" until secondApprovedBy fills in.
   */
  approvedBy: string | null;
  secondApprovedBy: string | null;
  payoutAccount: PayoutAccountSnapshot;
}

export const ADMIN_PAYOUT_STATUS_LABEL: Record<AdminPayoutStatus, string> = {
  pending_review: "Pending review",
  approved: "Approved",
  processing: "Processing",
  held: "Held",
  paid: "Paid",
  failed: "Failed",
  returned: "Returned",
  rejected: "Rejected",
  cancelled: "Cancelled",
};

export function payoutMethodLabel(method: PayoutMethod): string {
  return PAYOUT_METHOD_LABEL[method];
}
