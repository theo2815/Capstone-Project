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
  paymentMethod: string;
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
  paymentMethod,
}: AdminRefundConfirmModalProps) {
  return (
    <AdminConfirmModal
      open={open}
      submitting={submitting}
      pendingLabel="Refunding…"
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
        {paymentMethod === "qrph"
          ? "After you approve, make the refund link in the PayMongo dashboard and send it here. PayMongo's dashboard can only refund the whole payment."
          : "The refund goes to the original payment method when the provider confirms it. The dispute then closes."}
      </p>
      <p className="font-sans text-sm text-slate">
        The runner is told if they still have an account, and the photographer is told the dispute was resolved. A
        finished refund cannot be undone: the sale is taken back from the photographer&apos;s
        earnings and, on a full refund, the runner&apos;s download of this photo ends.
      </p>
    </AdminConfirmModal>
  );
}
