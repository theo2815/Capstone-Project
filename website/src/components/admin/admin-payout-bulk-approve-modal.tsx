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
      title="Approve payouts"
      confirmLabel="Confirm approve"
    >
      <p className="font-sans text-sm text-ink-soft">
        Approve{" "}
        <span className="text-ink font-display text-base tnum">{count}</span>{" "}
        {count === 1 ? "payout" : "payouts"} totaling{" "}
        <span className="text-ink font-display text-base tnum">
          {formatPrice(totalAmount)}
        </span>
        ? Payouts enter the payment queue once all required approvals are recorded.
      </p>
      <p className="font-sans text-sm text-slate">
        Requests that need two admins stay pending until a different admin
        records the second approval. Approving a payout does not send funds.
      </p>
    </AdminConfirmModal>
  );
}
