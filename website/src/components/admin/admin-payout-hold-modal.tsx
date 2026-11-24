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
  { value: "chargeback_review", label: "Many payment disputes on your recent sales" },
  { value: "dispute_pending", label: "Open refund requests on photos in this payout" },
  { value: "compliance_review", label: "We need to review your account first" },
  { value: "method_mismatch", label: "Your payout account details do not match our records" },
  { value: "manual_check", label: "We are checking this payout by hand" },
  { value: "other", label: "Other (see note)" },
];
const NOTE_SEPARATOR = " — ";
// FU-9 (FT-33): the server refuses a reason over 255 characters, so the note leaves room for the longest label.
const NOTE_MAX = 255 - Math.max(...HOLD_REASONS.map((r) => r.label.length)) - NOTE_SEPARATOR.length;

// Hold-payout modal — used both for single-row and bulk hold actions.
// Same shape as suspend/reject — radio + note (NOTE_MAX) composed into the
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
      note.trim().length > 0 ? `${label}${NOTE_SEPARATOR}${note.trim()}` : label;
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
            will be paused. Each photographer reads the reason below exactly as
            it is written, with your note.
          </>
        ) : (
          <>
            This payout{" "}
            <span className="text-ink tnum">
              ({formatPrice(totalAmount)})
            </span>{" "}
            will be paused. The photographer reads the reason below exactly as
            it is written, with your note. You can approve or reject the payout
            later.
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
          onChange={(reason) => { setSelectedReason(reason); setAttempted(false); }}
        />
      </AdminFormSection>
      <AdminTextarea
        id="hold-note"
        label={selectedReason === "other" ? "Note (required, max 280)" : "Note (optional, max 280)"}
        error={attempted && noteMissing ? "Add a note: Other needs one." : null}
        value={note}
        onChange={setNote}
        maxLength={NOTE_MAX}
        placeholder="What context should the photographer see?"
        hint="Seen by: the photographer, exactly as written."
      />
      {!selectedReason && <p className="font-sans text-sm text-slate">Choose a reason to continue.</p>}
    </AdminFormModal>
  );
}
