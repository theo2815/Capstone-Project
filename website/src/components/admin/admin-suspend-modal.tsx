"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminSuspendModalProps {
  open: boolean;
  submitting?: boolean;
  onClose: () => void;
  /** Resolves false when the request failed, so the modal keeps its input and shows why. */
  onSubmit: (reason: string) => Promise<boolean | void>;
  photographerName: string;
}

const SUSPEND_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "watermark_violation", label: "Watermark or branding violates policy" },
  { value: "fraud_suspected", label: "Suspected fraudulent uploads" },
  { value: "chargeback_pattern", label: "Chargeback pattern flagged by payments" },
  { value: "complaint_volume", label: "Multiple unresolved runner complaints" },
  { value: "compliance_review", label: "Compliance review requested" },
  { value: "other", label: "Other (see note)" },
];

// Suspend-with-reason modal. Same shape as the verifications Reject modal —
// canned reasons + 280-char note. The composed string flows through
// useAdminUserActions().suspend(userId, reason). Suspension is reversible from
// the same action aside via the Unsuspend button.
export function AdminSuspendModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  photographerName,
}: AdminSuspendModalProps) {
  // AX-28: no reason starts picked, so a quick Enter never records one the admin did not choose.
  const [selectedReason, setSelectedReason] = useState("");
  const [note, setNote] = useState("");
  const [attempted, setAttempted] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!open) { setNote(""); setSelectedReason(""); setAttempted(false); setFailed(false); }
  }, [open]);

  const noteMissing = selectedReason === "other" && note.trim().length === 0;

  async function handleSubmit() {
    if (submitting) return;
    if (noteMissing) { setAttempted(true); return; }
    const label =
      SUSPEND_REASONS.find((r) => r.value === selectedReason)?.label ??
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
      pendingLabel="Suspending…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Suspend account"
      intro={`${photographerName} will be hidden from public listings and unable to upload until you unsuspend. The reason is logged.`}
      submitLabel="Suspend"
      submitDisabled={!selectedReason}
      error={failed ? "Couldn't finish this. Your reason and note are still here." : null}
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="suspend-reason"
          options={SUSPEND_REASONS}
          value={selectedReason}
          onChange={setSelectedReason}
        />
      </AdminFormSection>
      <AdminTextarea
        id="suspend-note"
        label={selectedReason === "other" ? "Note (required, max 280)" : "Note (optional, max 280)"}
        error={attempted && noteMissing ? "Add a note: Other needs one." : null}
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What context should the next reviewer see?"
      />
    </AdminFormModal>
  );
}
