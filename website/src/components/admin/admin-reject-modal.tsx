"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminRejectModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (reason: string) => void;
  photographerName: string;
}

const REJECTION_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "watermark_missing", label: "Watermark is missing or unclear" },
  { value: "handle_unavailable", label: "Public handle is reserved or taken" },
  { value: "brand_inappropriate", label: "Brand or bio violates guidelines" },
  { value: "cover_low_quality", label: "Cover image quality too low" },
  { value: "social_unverified", label: "Social profiles couldn't be verified" },
  { value: "other", label: "Other (see note)" },
];

// Shared reject-with-reason modal. Used from /admin/verifications row
// actions and from /admin/photographers/[handle] action aside. Composes
// label + optional 280-char note into the reason string passed to
// useAdminUserStore.reject(userId, reason).
export function AdminRejectModal({
  open,
  onClose,
  onSubmit,
  photographerName,
}: AdminRejectModalProps) {
  const [selectedReason, setSelectedReason] = useState<string>(
    REJECTION_REASONS[0]!.value,
  );
  const [note, setNote] = useState("");

  function handleSubmit() {
    const label =
      REJECTION_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    onSubmit(composed);
    setNote("");
    setSelectedReason(REJECTION_REASONS[0]!.value);
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Send back for fixes"
      intro={`${photographerName} will see your reason on their settings page so they know what to update before resubmitting.`}
      submitLabel="Send back"
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="rejection-reason"
          options={REJECTION_REASONS}
          value={selectedReason}
          onChange={setSelectedReason}
        />
      </AdminFormSection>
      <AdminTextarea
        id="reject-note"
        label="Note (optional, max 280)"
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What would help them fix it faster?"
      />
    </AdminFormModal>
  );
}
