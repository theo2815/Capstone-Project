"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminSuspendModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (reason: string) => void;
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
// useAdminUserStore.suspend(userId, reason). Suspension is reversible from
// the same action aside via the Unsuspend button.
export function AdminSuspendModal({
  open,
  onClose,
  onSubmit,
  photographerName,
}: AdminSuspendModalProps) {
  const [selectedReason, setSelectedReason] = useState<string>(
    SUSPEND_REASONS[0]!.value,
  );
  const [note, setNote] = useState("");

  function handleSubmit() {
    const label =
      SUSPEND_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    onSubmit(composed);
    setNote("");
    setSelectedReason(SUSPEND_REASONS[0]!.value);
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Suspend account"
      intro={`${photographerName} will be hidden from public listings and unable to upload until you unsuspend. The reason is logged.`}
      submitLabel="Suspend"
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
        label="Note (optional, max 280)"
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What context should the next reviewer see?"
      />
    </AdminFormModal>
  );
}
