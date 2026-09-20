"use client";

import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import { formatPrice } from "@/lib/utils";

interface AdminRefundConfirmModalProps {
  open: boolean;
  submitting?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  amount: number;
  runnerHandle: string;
  isFull: boolean;
}

// Stacked confirmation child for the dispute Resolve flow. Mounts on top
// of <AdminResolveDisputeModal> with the parent's escDisabled set true so
// ESC closes only this. Cancel returns to the parent with its form state
// intact (state lives in the action-aside, not in either modal).
export function AdminRefundConfirmModal({
  open,
  submitting = false,
  onCancel,
  onConfirm,
  amount,
  runnerHandle,
  isFull,
}: AdminRefundConfirmModalProps) {
  return (
    <AdminConfirmModal
      open={open}
      submitting={submitting}
      onClose={onCancel}
      onConfirm={onConfirm}
      title="Confirm refund"
      cancelLabel="Back"
      confirmLabel="Confirm refund"
    >
      <p className="font-sans text-sm text-ink-soft">
        Refund{" "}
        <span className="text-ink font-display text-base tnum">
          {formatPrice(amount)}
        </span>{" "}
        ({isFull ? "full" : "partial"}) to{" "}
        <span className="text-ink">@{runnerHandle}</span>?
      </p>
      <p className="font-sans text-sm text-slate">
        The refund goes to the original payment method when the provider confirms it.
        The dispute then closes.
      </p>
    </AdminConfirmModal>
  );
}
