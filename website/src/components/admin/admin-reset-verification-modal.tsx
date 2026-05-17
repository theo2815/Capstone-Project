"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminResetVerificationModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (reason: string) => void;
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
  onClose,
  onSubmit,
  photographerName,
}: AdminResetVerificationModalProps) {
  const [selectedReason, setSelectedReason] = useState<string>(
    RESET_REASONS[0]!.value,
  );
  const [note, setNote] = useState("");

  function handleSubmit() {
    const label =
      RESET_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    onSubmit(composed);
    setNote("");
    setSelectedReason(RESET_REASONS[0]!.value);
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Reset verification"
      intro={`${photographerName} will drop back to incomplete and see your reason in their inbox so they know what to update before resubmitting.`}
      submitLabel="Reset verification"
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
        label="Note (optional, max 280)"
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="Anything specific they need to fix before resubmitting?"
      />
    </AdminFormModal>
  );
}
