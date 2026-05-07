"use client";

import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import { formatPrice } from "@/lib/utils";

interface AdminPayoutBulkApproveModalProps {
  open: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  count: number;
  totalAmount: number;
}

// Bulk-approve confirmation. No note field — approval is a positive
// action that doesn't need explanation. The mock loop just transitions
// each cycle's status; Phase F backend will atomically batch the same set
// of payout-id mutations.
export function AdminPayoutBulkApproveModal({
  open,
  onCancel,
  onConfirm,
  count,
  totalAmount,
}: AdminPayoutBulkApproveModalProps) {
  return (
    <AdminConfirmModal
      open={open}
      onClose={onCancel}
      onConfirm={onConfirm}
      title="Approve cycles"
      confirmLabel="Confirm approve"
    >
      <p className="font-sans text-sm text-ink-soft">
        Approve{" "}
        <span className="text-ink font-display text-base tnum">{count}</span>{" "}
        {count === 1 ? "cycle" : "cycles"} totaling{" "}
        <span className="text-ink font-display text-base tnum">
          {formatPrice(totalAmount)}
        </span>
        ? Approved cycles enter the queue for marking paid.
      </p>
      <p className="font-sans text-sm text-slate">
        Photographers see their cycle move to{" "}
        <span className="text-ink">Approved</span> on /dashboard/billing.
        Funds are not released until each cycle is marked paid with a
        reference number.
      </p>
    </AdminConfirmModal>
  );
}
