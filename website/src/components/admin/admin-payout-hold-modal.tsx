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
  submitting?: boolean;
  onClose: () => void;
  /** Resolves false when the request failed, so the modal keeps its input and shows why. */
  onSubmit: (reason: string) => Promise<boolean | void>;
  /** Payout ids being held (1 for single-row, n for bulk). */
  payoutIds: string[];
  /** Sum of amounts being held — shown in copy. */
  totalAmount: number;
}

const HOLD_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "chargeback_review", label: "Chargeback rate elevated this period" },
  { value: "dispute_pending", label: "Active disputes referencing this payout" },
  { value: "compliance_review", label: "Compliance review requested" },
  { value: "method_mismatch", label: "Payout method on file doesn't match record" },
  { value: "manual_check", label: "Manual review pending payments lead" },
  { value: "other", label: "Other (see note)" },
];

// Hold-payout modal — used both for single-row and bulk hold actions.
// Same shape as suspend/reject — radio + 280-char note composed into the
// reason string passed to usePayoutActions().hold or .bulkHold.
export function AdminPayoutHoldModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  payoutIds,
  totalAmount,
}: AdminPayoutHoldModalProps) {
  // AX-28: no reason starts picked, so a quick Enter never sends the photographer one the admin did not choose.
  // The caller mounts this modal only while a hold is open, so its state starts fresh each time.
  const [selectedReason, setSelectedReason] = useState("");
  const [note, setNote] = useState("");
  const [attempted, setAttempted] = useState(false);
  const [failed, setFailed] = useState(false);

  const isBulk = payoutIds.length > 1;
  const titleSuffix = isBulk ? `${payoutIds.length} payouts` : payoutIds[0] ?? "";
  const noteMissing = selectedReason === "other" && note.trim().length === 0;

  async function handleSubmit() {
    if (submitting) return;
    if (noteMissing) { setAttempted(true); return; }
    const label =
      HOLD_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    setFailed(false);
    setFailed((await onSubmit(composed)) === false);
  }

  return (
    <AdminFormModal
      tone="danger"
      open={open}
      submitting={submitting}
      pendingLabel="Holding…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Hold ${titleSuffix}`}
      intro={
        isBulk ? (
          <>
            <span className="text-ink tnum">{payoutIds.length}</span> payouts
            totaling{" "}
            <span className="text-ink tnum">{formatPrice(totalAmount)}</span>{" "}
            will be paused. Photographers are notified with the reason.
          </>
        ) : (
          <>
            This payout{" "}
            <span className="text-ink tnum">
              ({formatPrice(totalAmount)})
            </span>{" "}
            will be paused. The photographer is notified with the reason.
          </>
        )
      }
      submitLabel={`Hold ${isBulk ? `${payoutIds.length} payouts` : "payout"}`}
      submitDisabled={!selectedReason}
      error={failed ? "Couldn't finish this. Your reason and note are still here." : null}
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
        label={selectedReason === "other" ? "Note (required, max 280)" : "Note (optional, max 280)"}
        error={attempted && noteMissing ? "Add a note: Other needs one." : null}
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What context should the photographer see?"
      />
    </AdminFormModal>
  );
}
