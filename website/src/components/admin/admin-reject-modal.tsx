"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminRejectModalProps {
  open: boolean;
  submitting?: boolean;
  onClose: () => void;
  /** Resolves false when the request failed, so the modal keeps its input and shows why. */
  onSubmit: (reason: string) => Promise<boolean | void>;
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
// useAdminUserActions().reject(userId, reason).
export function AdminRejectModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  photographerName,
}: AdminRejectModalProps) {
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
      REJECTION_REASONS.find((r) => r.value === selectedReason)?.label ??
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
      pendingLabel="Sending back…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Send back for fixes"
      intro={`${photographerName} will see your reason on their settings page so they know what to update before resubmitting.`}
      submitLabel="Send back"
      submitDisabled={!selectedReason}
      error={failed ? "Couldn't finish this. Your reason and note are still here." : null}
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
        label={selectedReason === "other" ? "Note (required, max 280)" : "Note (optional, max 280)"}
        error={attempted && noteMissing ? "Add a note: Other needs one." : null}
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What would help them fix it faster?"
      />
    </AdminFormModal>
  );
}
