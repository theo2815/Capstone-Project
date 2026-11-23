"use client";

import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import type { AdminPayoutCycle } from "@/lib/admin-payouts";
import { formatPrice } from "@/lib/utils";

interface AdminPayoutBulkApproveModalProps {
  open: boolean;
  submitting?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  count: number;
  totalAmount: number;
  /** The payouts being approved, listed so the admin confirms people, not only a total. */
  cycles: AdminPayoutCycle[];
  /** The payouts a partial approve could not approve, with the server's reasons. */
  error?: string | null;
}

// Bulk-approve confirmation. No note field — approval is a positive
// action that doesn't need explanation. Per-cycle outcomes come from the
// backend; a successful first approval can still require a second admin.
export function AdminPayoutBulkApproveModal({
  open,
  submitting = false,
  onCancel,
  onConfirm,
  count,
  totalAmount,
  cycles,
  error = null,
}: AdminPayoutBulkApproveModalProps) {
  return (
    <AdminConfirmModal
      open={open}
      submitting={submitting}
      pendingLabel="Approving…"
      onClose={onCancel}
      onConfirm={onConfirm}
      title="Approve payouts"
      confirmLabel="Approve"
      error={error}
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
      {/* Five lines show, then the list scrolls. */}
      <ul className="max-h-[12.7rem] overflow-y-auto rounded-xl border border-line divide-y divide-line">
        {cycles.map((cycle) => (
          <li key={cycle.id} className="flex items-baseline justify-between gap-3 px-3 py-2 font-sans text-sm">
            <span className="min-w-0 truncate">
              <span className="font-mono text-ink">{cycle.id}</span>{" "}
              <span className="text-ink-soft">{cycle.brandName ?? cycle.photographerName}</span>
            </span>
            <span className="shrink-0 text-ink tnum">{formatPrice(cycle.amount)}</span>
          </li>
        ))}
      </ul>
      <p className="font-sans text-sm text-slate">
        Approving sends no money. A larger payout needs a second, different
        admin before it can be paid.
      </p>
    </AdminConfirmModal>
  );
}
