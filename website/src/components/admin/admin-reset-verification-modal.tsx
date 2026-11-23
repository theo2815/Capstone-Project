"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminResetVerificationModalProps {
  open: boolean;
  submitting?: boolean;
  onClose: () => void;
  /** Resolves false when the request failed, so the modal keeps its input and shows why. */
  onSubmit: (reason: string) => Promise<boolean | void>;
  photographerName: string;
}

const RESET_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "policy_violation", label: "Policy violation surfaced after approval" },
  { value: "watermark_changed", label: "Watermark or brand no longer matches what was approved" },
  { value: "handle_dispute", label: "Public handle dispute or change required" },
  { value: "stale_verification", label: "Verification info is stale — needs refresh" },
  { value: "other", label: "Other (see note)" },
];

// Reset-verification reason modal. Mirrors AdminRejectModal but for the
// reset action — drops an approved photographer back to "incomplete" so
// they re-submit. The composed reason lands in their inbox so they know
// what to update before re-submitting.
export function AdminResetVerificationModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  photographerName,
}: AdminResetVerificationModalProps) {
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
      RESET_REASONS.find((r) => r.value === selectedReason)?.label ??
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
      pendingLabel="Resetting verification…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Reset verification"
      intro={`${photographerName} goes back to Incomplete and gets your reason in their inbox. Until they send their profile again and you approve it, they cannot upload and their public page does not open. Photos already sold keep working.`}
      submitLabel="Reset verification"
      submitDisabled={!selectedReason}
      error={failed ? "Couldn't finish this. Your reason and note are still here." : null}
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="reset-reason"
          options={RESET_REASONS}
          value={selectedReason}
          onChange={setSelectedReason}
        />
      </AdminFormSection>
      <AdminTextarea
        id="reset-note"
        label={selectedReason === "other" ? "Note (required, max 280)" : "Note (optional, max 280)"}
        error={attempted && noteMissing ? "Add a note: Other needs one." : null}
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="Anything specific they need to fix before resubmitting?"
      />
    </AdminFormModal>
  );
}
