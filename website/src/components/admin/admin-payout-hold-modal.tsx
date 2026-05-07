"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";
import { formatPrice } from "@/lib/utils";

interface AdminPayoutHoldModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (reason: string) => void;
  /** Cycle ids being held (1 for single-row, n for bulk). */
  payoutIds: string[];
  /** Sum of amounts being held — shown in copy. */
  totalAmount: number;
}

const HOLD_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "chargeback_review", label: "Chargeback rate elevated this cycle" },
  { value: "dispute_pending", label: "Active disputes referencing this cycle" },
  { value: "compliance_review", label: "Compliance review requested" },
  { value: "method_mismatch", label: "Payout method on file doesn't match record" },
  { value: "manual_check", label: "Manual review pending payments lead" },
  { value: "other", label: "Other (see note)" },
];

// Hold-cycle modal — used both for single-row and bulk hold actions.
// Same shape as suspend/reject — radio + 280-char note composed into the
// reason string passed to useAdminPayoutStore.hold or .bulkHold.
export function AdminPayoutHoldModal({
  open,
  onClose,
  onSubmit,
  payoutIds,
  totalAmount,
}: AdminPayoutHoldModalProps) {
  const [selectedReason, setSelectedReason] = useState<string>(
    HOLD_REASONS[0]!.value,
  );
  const [note, setNote] = useState("");

  const isBulk = payoutIds.length > 1;
  const titleSuffix = isBulk ? `${payoutIds.length} cycles` : payoutIds[0] ?? "";

  function handleSubmit() {
    const label =
      HOLD_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    onSubmit(composed);
    setNote("");
    setSelectedReason(HOLD_REASONS[0]!.value);
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Hold ${titleSuffix}`}
      intro={
        isBulk ? (
          <>
            <span className="text-ink tnum">{payoutIds.length}</span> cycles
            totaling{" "}
            <span className="text-ink tnum">{formatPrice(totalAmount)}</span>{" "}
            will be paused. Photographers are notified with the reason.
          </>
        ) : (
          <>
            This cycle{" "}
            <span className="text-ink tnum">
              ({formatPrice(totalAmount)})
            </span>{" "}
            will be paused. The photographer is notified with the reason.
          </>
        )
      }
      submitLabel={`Hold ${isBulk ? `${payoutIds.length} cycles` : "cycle"}`}
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="hold-reason"
          options={HOLD_REASONS}
          value={selectedReason}
          onChange={setSelectedReason}
        />
      </AdminFormSection>
      <AdminTextarea
        id="hold-note"
        label="Note (optional, max 280)"
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What context should the photographer see?"
      />
    </AdminFormModal>
  );
}
